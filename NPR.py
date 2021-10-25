import cv2
import argparse
from PIL import Image

def NPR(src):
    epf = cv2.edgePreservingFilter(src, flags=1, sigma_s=60, sigma_r=0.4)
    de = cv2.detailEnhance(src, sigma_s=10, sigma_r=0.15)
    pen_gray, pen_col = cv2.pencilSketch(src, sigma_s=60, sigma_r=0.1, shade_factor=0.03)
    style = cv2.stylization(src, sigma_s=60, sigma_r=0.07)
    return epf, de, pen_col, style

def webcam_or_pic2npr(out,is_webcam,pic):
    if is_webcam:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            e,d,p,s = NPR(frame)
            cv2.imshow('raw_input', frame)
            cv2.imshow('edgePreservingFilter',e)
            cv2.imshow('detailEnhance',d)
            cv2.imshow('pencilSketch',p)
            cv2.imshow('stylization',s)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(out,frame)
                cv2.imwrite(out,p)
        cap.release()
        cv2.destroyAllWindows()
    else:
        img = cv2.imread(pic, cv2.IMREAD_COLOR)
        e,d,p,s = NPR(img)
        cv2.imwrite(str(out)+'edgePreservingFilter.png',e)
        cv2.imwrite(str(out)+'detailEnhance.png',d)
        cv2.imwrite(str(out)+'pencilSketch.png',p)
        cv2.imwrite(str(out)+'stylization.png',s)

def main():
    parser = argparse.ArgumentParser(description='python+opencv_npr')
    parser.add_argument('--in_pic','-i',default='sample.png',help='input_picture_name')
    parser.add_argument('--out','-o',default='./',help='output_dir')
    parser.add_argument('--is_webcam',action='store_true',help='use webwebcam_or_pic2npr')
    args = parser.parse_args()
    webcam_or_pic2npr(args.out, args.is_webcam, args.in_pic)

if __name__ == "__main__":
    main()