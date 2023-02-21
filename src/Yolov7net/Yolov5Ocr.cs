using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Concurrent;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Yolov7net.Extentions;
using Yolov7net.Models;

namespace Yolov7net
{
    /// <summary>
    /// yolov5、yolov6 模型,不包含nms结果
    /// </summary>
    public class Yolov5Ocr : IDisposable
    {
        private readonly InferenceSession _inferenceSession;
        private YoloModelOcr _model = new YoloModelOcr();
        private string _inputName = "";
        private static string PLATE_NAME =
           "#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新" +
           "学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品";

        private static float STD_VALUE = 0.193f;
        private static float MEAN_VALUE = 0.588f * 255;
        // clors = [(255,0,0) 红       ,(0,255,0)绿     ,(0,0,255)蓝     ,(255,255,0)黄     ,(0,255,255)青]
        private static string[] PLATE_COLOR = new String[] { "黑色", "蓝色", "绿色", "白色", "黄色" };



        public Yolov5Ocr(string ModelPath, bool useCuda = false, string inputName = "images")
        {

            if (useCuda)
            {
                SessionOptions opts = SessionOptions.MakeSessionOptionWithCudaProvider();
                _inferenceSession = new InferenceSession(ModelPath, opts);
            }
            else
            {
                SessionOptions opts = new();
                _inferenceSession = new InferenceSession(ModelPath, opts);
            }

            _inputName = inputName;
            /// Get model info
            get_input_details();
            get_output_details();
        }

        public void SetupLabels(string[] labels)
        {
            labels.Select((s, i) => new { i, s }).ToList().ForEach(item =>
            {
                _model.Labels.Add(new YoloLabel { Id = item.i, Name = item.s });
            });
        }

        public void SetupYoloDefaultLabels()
        {
            var s = new string[] { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
            SetupLabels(s);
        }

        public string[] Predict(Image image, float conf_thres = 0, float iou_thres = 0)
        {
            if (conf_thres > 0f)
            {
                _model.Confidence = conf_thres;
                _model.MulConfidence = conf_thres + 0.05f;
            }
            if (iou_thres > 0f)
            {
                _model.Overlap = iou_thres;
            }
            return ParseOutput(Inference(image), image);
        }

        /// <summary>
        /// Removes overlaped duplicates (nms).
        /// </summary>
        private List<YoloPrediction> Supress(List<YoloPrediction> items)
        {
            var result = new List<YoloPrediction>(items);

            foreach (var item in items) // iterate every prediction
            {
                foreach (var current in result.ToList()) // make a copy for each iteration
                {
                    if (current == item) continue;

                    var (rect1, rect2) = (item.Rectangle, current.Rectangle);

                    RectangleF intersection = RectangleF.Intersect(rect1, rect2);

                    float intArea = intersection.Area(); // intersection area
                    float unionArea = rect1.Area() + rect2.Area() - intArea; // union area
                    float overlap = intArea / unionArea; // overlap ratio

                    if (overlap >= _model.Overlap)
                    {
                        if (item.Score >= current.Score)
                        {
                            result.Remove(current);
                        }
                    }
                }
            }

            return result;
        }

        private DenseTensor<float>[] Inference(Image img)
        {
            Bitmap resized = null;

            if (img.Width != _model.Width || img.Height != _model.Height)
            {
                resized = Utils.ResizeImage(img, _model.Width, _model.Height); // fit image size to specified input size
            }
            else
            {
                resized = new Bitmap(img);
            }

            var inputs = new List<NamedOnnxValue> // add image as onnx input
            {              

              NamedOnnxValue.CreateFromTensor(_inputName, Utils.ExtractPixels(resized))
            };


            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result = _inferenceSession.Run(inputs); // run inference

            //  _inferenceSession.ge

            var output = new List<DenseTensor<float>>();

            foreach (var item in _model.Outputs) // add outputs for processing
            {
                output.Add(result.First(x => x.Name == item).Value as DenseTensor<float>);



            };

            //var s = maxScoreIndex(ouyt[0]);
            return output.ToArray();
        }
        /*-----------------*/


        private string[] ParseOutput(DenseTensor<float>[] output, Image image)
        {
            int allValues = output[0].Count();
            int[] dims = output[0].Dimensions.ToArray();
            int crnnRows = allValues / dims[2];
            var resultsTxt = ScoreToTextLine(output[0].AsEnumerable<float>().ToArray(), crnnRows, dims[2]);
            
            var colorA = output[1].AsEnumerable<float>().ToArray();
            var colorN = "";
           
            double index = -1;
            double max = double.MinValue;
            for (int i = 0; i < colorA.Length; i++)
            {
                if (max < colorA[i])
                {
                    max = colorA[i];
                    index = i;
                }
            }
            colorN = PLATE_COLOR[(int)index];

            var back = new string[] { resultsTxt, colorN };
            return back;
            //  return _model.UseDetect ? ParseDetect(output[0], image) : ParseSigmoid(output, image);
        }


        /// <summary>
        /// 解析文字
        /// </summary>
        /// <param name="srcData"></param>
        /// <param name="rows"></param>
        /// <param name="cols"></param>
        /// <returns></returns>
        private string ScoreToTextLine(float[] srcData, int rows, int cols)
        {
            StringBuilder sb = new StringBuilder();
            // TextLine textLine = new TextLine();

            int lastIndex = 0;
            List<float> scores = new List<float>();

            for (int i = 0; i < rows; i++)
            {
                int maxIndex = 0;
                float maxValue = -1000F;

                //do softmax
                List<float> expList = new List<float>();
                for (int j = 0; j < cols; j++)
                {
                    float expSingle = (float)Math.Exp(srcData[i * cols + j]);
                    expList.Add(expSingle);
                }
                float partition = expList.Sum();
                for (int j = 0; j < cols; j++)
                {
                    float softmax = expList[j] / partition;
                    if (softmax > maxValue)
                    {
                        maxValue = softmax;
                        maxIndex = j;
                    }
                }

                if (maxIndex > 0 && maxIndex < PLATE_NAME.Length && (!(i > 0 && maxIndex == lastIndex)))
                {
                    scores.Add(maxValue);
                    sb.Append(PLATE_NAME[maxIndex]);
                }
                lastIndex = maxIndex;
            }

            return sb.ToString();
        }


        private void get_input_details()
        {
            _model.Height = _inferenceSession.InputMetadata[_inputName].Dimensions[2];
            _model.Width = _inferenceSession.InputMetadata[_inputName].Dimensions[3];
        }

        private void get_output_details()
        {
            _model.Outputs = _inferenceSession.OutputMetadata.Keys.ToArray();
            _model.Dimensions = _inferenceSession.OutputMetadata[_model.Outputs[0]].Dimensions[2];
            _model.UseDetect = !(_model.Outputs.Any(x => x == "score"));
        }

        public void Dispose()
        {
            _inferenceSession.Dispose();
        }
    }
}