using OpenCvSharp;
using OpenCvSharp.Extensions;
using System.Drawing;
using System.Drawing.Imaging;
using Yolov7net;

namespace Yolov7Ml
{
    internal class Program
    {
        static void Main(string[] args)
        {


            /*-------------------------定义检测范围---------------------------------------------*/
            using var yolo = new Yolov5("./assets/plate_detect_v7.onnx", inputName: "input"); //yolov7 e2e 模型,不需要 nms 操作
            /// using var yolo = new Yolov5("./assets/plate_detect.onnx", inputName: "input"); //yolov7 e2e 模型,不需要 nms 操作
                                                                                           // setup labels of onnx model 

            yolo.SetupYoloDefaultLabels(); // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)
            var dirPos = Directory.GetFiles(@"./Assets/input");
            var dtNow = DateTime.Now.Hour.ToString() + DateTime.Now.Minute.ToString() + DateTime.Now.Second.ToString();

            /*-------------------------定义识别结果---------------------------------------------*/
            using var yoloOcr = new Yolov5Ocr("./assets/plate_rec_color.onnx", inputName: "images"); //yolov7 e2e 模型,不需要 nms 操作
                                                                                                     // setup labels of onnx model 

            yoloOcr.SetupYoloDefaultLabels(); // use custom trained model should use your labels like: yolo.SetupLabels(string[] labels)

            OctTest octTest = new OctTest();


            foreach (var item in dirPos)
            {
                if (Path.GetExtension(item).Equals(".jpg") ||
                    Path.GetExtension(item).Equals(".png") ||
                    Path.GetExtension(item).Equals(".jpeg"))
                {
                    using var image = System.Drawing.Image.FromFile(item);
                    var predictions = yolo.Predict(image);

                    if (predictions.Count < 1) continue;

                    var fileName = Path.GetFileName(item);
                    var path = Path.GetDirectoryName(item);
                    using var graphics = Graphics.FromImage(image);
                    int num = 0;
                    foreach (var prediction in predictions) // iterate predictions to draw results
                    {
                        num++;

                        double score = Math.Round(prediction.Score, 2);
                        var labelRect = prediction.Rectangle;
                        var twoLayers = (labelRect.Height / labelRect.Width) > 0.5;

                        //定义截取矩形
                        System.Drawing.Rectangle cropArea = new System.Drawing.Rectangle((int)labelRect.X < 0 ? 0 : (int)labelRect.X, (int)labelRect.Y < 0 ? 0 : (int)labelRect.Y, (int)labelRect.Width, (int)labelRect.Height);
                        //定义Bitmap对象
                        System.Drawing.Bitmap bmpImage = new System.Drawing.Bitmap(image);
                        //进行裁剪
                        System.Drawing.Bitmap bmpCrop = bmpImage.Clone(cropArea, bmpImage.PixelFormat);
                        //保存成新文件
                        // bmpCrop.Save(Path.Combine(path, (fileName + "_" + dtNow + "_clone.png")), ImageFormat.Png);

                    

                        if (twoLayers)
                        {
                            var img_upper_H = labelRect.Height / 2;

                            var width = (int)labelRect.Width + 1;
                            var height = (int)img_upper_H;
                            Bitmap resultBitmap = new Bitmap(width, height);
                            using (Graphics g = Graphics.FromImage(resultBitmap))
                            {
                                Rectangle resultRectangle = new Rectangle(0, 0, width, height);
                                Rectangle sourceRectangle = new Rectangle(0, 0, width, height);
                                g.DrawImage(bmpCrop, resultRectangle, sourceRectangle, GraphicsUnit.Pixel);
                            }

                            Bitmap resultBitmap1 = new Bitmap(width, height);
                            using (Graphics g = Graphics.FromImage(resultBitmap1))
                            {
                                Rectangle resultRectangle = new Rectangle(0, 0, width, height);
                                Rectangle sourceRectangle = new Rectangle(0, height, width, height);
                                g.DrawImage(bmpCrop, resultRectangle, sourceRectangle, GraphicsUnit.Pixel);
                            }

                            bmpCrop = JoinImage(resultBitmap, resultBitmap1);


                        }
                        //保存成新文件
                        bmpCrop.Save(Path.Combine(path, (fileName + "_" + dtNow + num + "_clone.png")), ImageFormat.Png);

                        var backtxt = "";
                        //using (MemoryStream stream = new MemoryStream())
                        //{
                        //    bmpCrop.Save(stream, ImageFormat.Jpeg);
                        //    byte[] data = new byte[stream.Length];
                        //    stream.Seek(0, SeekOrigin.Begin);
                        //    stream.Read(data, 0, Convert.ToInt32(stream.Length));
                        //    // backtxt = ocrTxt.GetTest(data);
                        //}



                        var yoloOcrpredictions = yoloOcr.Predict(bmpCrop);
                        if (yoloOcrpredictions.Length > 0)
                        {
                            backtxt = "(车牌颜色：" + yoloOcrpredictions[1] + ")(车牌号：" + yoloOcrpredictions[0] + ")";
                        }

                        graphics.DrawRectangles(new Pen(prediction.Label.Color, 1), new[] { prediction.Rectangle });

                        var (x, y) = (prediction.Rectangle.X, prediction.Rectangle.Y + (labelRect.Height / 2));

                        graphics.DrawString($"{prediction.Label.Name} ({score})({backtxt})",
                                        new Font("Consolas", 14, GraphicsUnit.Pixel), new SolidBrush(Color.Red),
                                        new PointF(0, y));
                    }


                    image.Save(Path.Combine(path, (fileName + "_" + dtNow + ".png")), ImageFormat.Png);
                }
            }


            Console.WriteLine("Hello, World!");
        }


        /// <summary>
        /// 图片拼接
        /// </summary>
        /// <param name="sourceImg">图片1</param>
        /// <param name="newImg">图片2</param>
        /// <returns>拼接后的图片</returns>
        private static Bitmap JoinImage(Image sourceImg, Image newImg)
        {
            int imgHeight = 0, imgWidth = 0;

            imgWidth = sourceImg.Width + newImg.Width;
            imgHeight = sourceImg.Height > newImg.Height ? sourceImg.Height : newImg.Height;

            Bitmap joinedBitmap = new Bitmap(imgWidth, imgHeight);
            using (Graphics graph = Graphics.FromImage(joinedBitmap))
            {
                graph.DrawImage(sourceImg, 0, 0, sourceImg.Width, sourceImg.Height);

                graph.DrawImage(newImg, sourceImg.Width, 0, newImg.Width, newImg.Height);
            }
            return joinedBitmap;
        }
    }
}