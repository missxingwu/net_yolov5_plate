
using Sdcb.PaddleInference;
using Sdcb.PaddleOCR.Models.LocalV3;
using Sdcb.PaddleOCR.Models;
using Sdcb.PaddleOCR;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Net.WebRequestMethods;
using OpenCvSharp;
using Microsoft.Win32.SafeHandles;

namespace Yolov7Ml
{
    public class OctTest
    {
        FullOcrModel model = LocalFullModels.ChineseV3;


        public string GetColor(Bitmap image)
        {


            Mat src = OpenCvSharp.Extensions.BitmapConverter.ToMat(image);


            Mat hsv = new Mat();
            Cv2.CvtColor(src, hsv, ColorConversionCodes.BGR2HSV);

            Mat mask = new Mat();
            Cv2.InRange(hsv, new Scalar(35, 43, 46), new Scalar(77, 255, 255), mask);
            bool green = false;
            for (int r = 0; r < hsv.Rows; r++)
            {
                for (int c = 0; c < hsv.Cols; c++)
                {
                    if (mask.At<byte>(r, c) == 255)
                    {
                        green = true;
                    }
                }
            }

            Dictionary<string, Scalar[]> hsv_color_list = new System.Collections.Generic.Dictionary<string, Scalar[]>();
            hsv_color_list.Add("red", new Scalar[] { new Scalar(0, 43, 46), new Scalar(6, 255, 255) });
            hsv_color_list.Add("green", new Scalar[] { new Scalar(35, 43, 46), new Scalar(77, 255, 255) });
            hsv_color_list.Add("blue", new Scalar[] { new Scalar(100, 43, 46), new Scalar(124, 255, 255) });
            hsv_color_list.Add("black", new Scalar[] { new Scalar(0, 0, 0), new Scalar(180, 255, 46) });
            hsv_color_list.Add("white", new Scalar[] { new Scalar(0, 0, 221), new Scalar(180, 43, 220) });

            hsv_color_list.Add("yellow", new Scalar[] { new Scalar(26, 43, 46), new Scalar(34, 255, 255) });

            Dictionary<string, int> colorName = new Dictionary<string, int>();
            //OpenCvSharp.Size kernel = new OpenCvSharp.Size(3, 3);
            Mat kernel_mask = new Mat();
            //Cv2.Blur(hsv, kernel_mask, kernel);
            //图像B卷积核的定义
            InputArray kernel = Cv2.GetStructuringElement(MorphShapes.Rect, new OpenCvSharp.Size(20, 20), new OpenCvSharp.Point(7, 5));

            //  Cv2.Threshold(mask, kernel_mask, 0, 255, ThresholdTypes.Binary); //二值化
            //1.腐蚀
            Cv2.Erode(mask, kernel_mask, kernel);

            Cv2.Dilate(kernel_mask, kernel_mask, kernel);

            foreach (var item in hsv_color_list)
            {
                using (Mat color_mask = new Mat())
                {
                    int num = 0;
                    Cv2.InRange(kernel_mask, item.Value[0], item.Value[1], color_mask);

                    for (int r = 0; r < hsv.Rows; r++)
                    {
                        for (int c = 0; c < hsv.Cols; c++)
                        {
                            if (color_mask.At<byte>(r, c) == 255)
                            {
                                num++;
                            }
                        }
                    }

                    colorName.Add(item.Key, num);
                }
            }



            var s = colorName.FirstOrDefault(x => x.Value == colorName.Values.Max());

            return s.Key;
        }
    }
}
