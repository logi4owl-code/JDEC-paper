import numpy as np
import cv2
import os
import argparse
import torch
from torchvision import transforms
import numpy as np
import models
import utils.custom_transforms as ctrans
import dct_manip as dm
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2):
    # img1 and img2 are numpy arrays (H, W, C)
    # skimage ssim expects (H, W, C) with multichannel=True or (H, W)
    return ssim(img1, img2, multichannel=True, data_range=255)

def main(args):
    # Setup paths
    input_image_path = args.input
    model_path = args.model
    output_dir = args.output
    quality = args.quality

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load Model
    print(f"Loading model from {model_path}...")
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        if 'model' in checkpoint:
            model_spec = checkpoint['model']
        else:
            model_spec = checkpoint # Handle case where it's just state dict or model directly?
            # Assuming 'model' key as per test.py

        # NOTE: Using models.make as per test.py
        if torch.cuda.is_available():
            model = models.make(model_spec, load_sd=True).cuda()
        else:
            model = models.make(model_spec, load_sd=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prepare Image
    print(f"Processing image {input_image_path} with Quality {quality}...")
    img_orig = cv2.imread(input_image_path)
    if img_orig is None:
        print("Error reading input image.")
        return

    # Save original (as PNG to preserve quality)
    original_save_path = os.path.join(output_dir, 'original.png')
    cv2.imwrite(original_save_path, img_orig)

    # Pre-process image (pad/resize if necessary?)
    # test.py does some flipping and concatenation, but that seems to be for data augmentation or specific size handling.
    # It creates a collage of mirrored versions.
    # "img_png_ = np.concatenate([img_png_, np.flip(img_png_, [0])], 0)..."
    # And then takes [:size, :size, :] where size=1120.
    # The model might expect specific block sizes.
    # batch_y rearrange suggests s1=140, s2=140? No, that's for patch processing likely.

    # Let's see models.py or assume model handles arbitrary sizes if fully convolutional?
    # However, JDEC seems to work on blocks/coefficients.
    # The test.py code forces the image to be 1120x1120 by mirroring/cropping.
    # If our input image is smaller/larger, we might need to be careful.
    # Lena is 512x512.

    # Let's try to run without the mirroring first, but ensure dimensions are multiples of something (e.g. 8 for JPEG).
    h, w, _ = img_orig.shape

    # Pad image to be compatible with model and JPEG structure
    # Must be multiple of 16 (JPEG 4:2:0 MCU) and 28 (Swin window alignment)
    # LCM(16, 28) = 112
    align = 112
    pad_h = (align - h % align) % align
    pad_w = (align - w % align) % align

    if pad_h > 0 or pad_w > 0:
        print(f"Padding image by ({pad_h}, {pad_w}) to match model requirements...")
        img_padded = cv2.copyMakeBorder(img_orig, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    else:
        img_padded = img_orig

    # Compress to JPEG
    temp_jpg_path = os.path.join(output_dir, 'temp_input.jpg')
    cv2.imwrite(temp_jpg_path, img_padded, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    # Save the compressed input for comparison (save the padded one if padded, or crop later?)
    # The requirement is to output "JPEG file". I will save the one used for inference.
    # However, for side-by-side comparison, we might want to crop back.
    compressed_save_path = os.path.join(output_dir, 'compressed.jpg')
    import shutil
    shutil.copy(temp_jpg_path, compressed_save_path)

    # Load coefficients using dct_manip
    input_coeffs = dm.read_coefficients(temp_jpg_path)
    # input_coeffs: [Y_coef, Q_tables, Y_swin, CbCr_swin] or similar?
    # test.py:
    # input_ = dm.read_coefficients('./bin/temp_.jpg')
    # inp_swin = input_[2]
    # inp_swin_cbcr = input_[3]
    # dqt_swin = input_[1]

    inp_swin = input_coeffs[2]
    inp_swin_cbcr = input_coeffs[3]
    dqt_swin = input_coeffs[1]

    q_y = dqt_swin[0]
    q_cbcr = dqt_swin[1]

    # Preprocessing
    inp_swin = torch.clamp(inp_swin * q_y, min=-1024, max=1016)
    inp_swin_cbcr = torch.clamp(inp_swin_cbcr * q_cbcr, min=-1024, max=1016)

    normalize = ctrans.ToRange(val_min=-1, val_max=1, orig_min=-1024, orig_max=1016)

    inp_swin = normalize(inp_swin)
    inp_swin_cbcr = normalize(inp_swin_cbcr)
    dqt_swin = torch.stack([q_y, q_cbcr], dim=0)
    dqt_swin = normalize(dqt_swin)

    # Inference
    print("Running inference...")
    with torch.no_grad():
        # Add batch dimension and move to cuda
        inp_swin_t = inp_swin.unsqueeze(0)
        inp_swin_cbcr_t = inp_swin_cbcr.unsqueeze(0)
        dqt_swin_t = dqt_swin.unsqueeze(0)

        if torch.cuda.is_available():
            inp_swin_t = inp_swin_t.cuda()
            inp_swin_cbcr_t = inp_swin_cbcr_t.cuda()
            dqt_swin_t = dqt_swin_t.cuda()

        pred = model(inp_swin_t, inp_swin_cbcr_t, dqt_swin_t)
        pred = pred.squeeze(0).detach().cpu() + 0.5 # De-normalize? test.py does +0.5

    # pred is (C, H, W) float tensor
    # Crop if necessary to original size (test.py crops to h, w)
    pred = pred[:, :h, :w]

    # Convert to numpy image
    pred_np = (pred * 255).round().clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
    # Convert RGB to BGR for OpenCV
    pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)

    # Save restored image
    restored_save_path = os.path.join(output_dir, 'restored.png')
    cv2.imwrite(restored_save_path, pred_bgr)

    # Metrics
    # Load compressed image as numpy for metric calculation (it has artifacts)
    img_compressed = cv2.imread(temp_jpg_path)
    # Crop compressed image to original size
    img_compressed = img_compressed[:h, :w, :]

    psnr_orig_restored = calculate_psnr(img_orig, pred_bgr)
    ssim_orig_restored = calculate_ssim(img_orig, pred_bgr)

    psnr_orig_compressed = calculate_psnr(img_orig, img_compressed)
    ssim_orig_compressed = calculate_ssim(img_orig, img_compressed)

    print(f"Compressed (Quality {quality}) - PSNR: {psnr_orig_compressed:.2f}, SSIM: {ssim_orig_compressed:.4f}")
    print(f"Restored                   - PSNR: {psnr_orig_restored:.2f}, SSIM: {ssim_orig_restored:.4f}")

    # Comparison Image (Side-by-Side)
    # Resize images to same height if needed (they are same size here)
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2

    def add_label(img, text):
        img_labelled = img.copy()
        cv2.putText(img_labelled, text, (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
        return img_labelled

    img_orig_lbl = add_label(img_orig, "Original")
    img_comp_lbl = add_label(img_compressed, f"JPEG (Q{quality})")
    img_rest_lbl = add_label(pred_bgr, "Restored")

    comparison_img = np.concatenate((img_orig_lbl, img_comp_lbl, img_rest_lbl), axis=1)
    comparison_save_path = os.path.join(output_dir, 'comparison_side_by_side.png')
    cv2.imwrite(comparison_save_path, comparison_img)

    # Difference Map
    # Calculate absolute difference between Original and Restored
    diff = cv2.absdiff(img_orig, pred_bgr)
    # Enhance difference for visibility (e.g., multiply by 5 or 10)
    diff_enhanced = cv2.multiply(diff, 5)

    # Invert for white background? Or heatmap?
    # Simple heatmap
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_heatmap = cv2.applyColorMap(diff_gray * 5, cv2.COLORMAP_JET)

    diff_save_path = os.path.join(output_dir, 'difference_map.png')
    cv2.imwrite(diff_save_path, diff_heatmap)

    # Combined with diff
    comparison_with_diff = np.concatenate((comparison_img, add_label(diff_heatmap, "Diff (x5)")), axis=1)
    comparison_all_path = os.path.join(output_dir, 'comparison_all.png')
    cv2.imwrite(comparison_all_path, comparison_with_diff)

    print("Verification complete. Outputs saved to:")
    print(f" - {original_save_path}")
    print(f" - {compressed_save_path}")
    print(f" - {restored_save_path}")
    print(f" - {comparison_save_path}")
    print(f" - {diff_save_path}")
    print(f" - {comparison_all_path}")

    # Remove temp file
    if os.path.exists(temp_jpg_path):
        os.remove(temp_jpg_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='verification_output', help='Output directory')
    parser.add_argument('--quality', type=int, default=30, help='JPEG Quality factor')
    args = parser.parse_args()
    main(args)
