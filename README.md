# GML_TEST2
To test deployment of keras saved_model to Google Cloud ML for serving

Keras model of inception v3 was saved as saved_model for serving on google ml-engine.
Saved model was deployed to cloud. Upon attempted test of model with json instance request, received following error msg:

`{
  "error": "Prediction failed: Error during model execution: AbortionError(code=StatusCode.INVALID_ARGU
MENT, details=\"NodeDef mentions attr 'dilations' not in Op<name=Conv2D; signature=input:T, filter:T ->
 output:T; attr=T:type,allowed=[DT_HALF, DT_FLOAT]; attr=strides:list(int); attr=use_cudnn_on_gpu:bool,
default=true; attr=padding:string,allowed=[\"SAME\", \"VALID\"]; attr=data_format:string,default=\"NHWC
\",allowed=[\"NHWC\", \"NCHW\"]>; NodeDef: import_1/conv2d_1/convolution = Conv2D[T=DT_FLOAT, _output_s
hapes=[[?,149,149,32]], data_format=\"NHWC\", dilations=[1, 1, 1, 1], padding=\"VALID\", strides=[1, 2,
 2, 1], use_cudnn_on_gpu=true, _device=\"/job:localhost/replica:0/task:0/device:CPU:0\"](import/ResizeB
ilinear, import_1/conv2d_1/kernel). (Check whether your GraphDef-interpreting binary is up to date with
 your GraphDef-generating binary.).\n\t [[Node: import_1/conv2d_1/convolution = Conv2D[T=DT_FLOAT, _out
put_shapes=[[?,149,149,32]], data_format=\"NHWC\", dilations=[1, 1, 1, 1], padding=\"VALID\", strides=[
1, 2, 2, 1], use_cudnn_on_gpu=true, _device=\"/job:localhost/replica:0/task:0/device:CPU:0\"](import/Re
sizeBilinear, import_1/conv2d_1/ke...TRUNCATED\")"
}'
