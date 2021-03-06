?	??R??@??R??@!??R??@	?JTW??@?JTW??@!?JTW??@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??R??@a6?????1??f??@A?u?;O<??I\;Q)@YLU?????rEagerKernelExecute 0*	cX9??b@2F
Iterator::Model???
?ͭ?!?\t;?EC@)?&?%????1YL8n?9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?d ??ƥ?!?}O??)<@)wd?6????1;J.+8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa?^Cp\??!؀?QA?<@)?g????1j?a?3@:Preprocessing2U
Iterator::Model::ParallelMapV2????߽??!??8}?)@)????߽??1??8}?)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicej?? ?m??!???5?#@)j?? ?m??1???5?#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???1v·?!???F?N@)?R]??{?1??_?}@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????x?!b????@)????x?1b????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?34.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?JTW??@Iӎ?GI@Q???ؚF@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	a6?????a6?????!a6?????      ??!       "	??f??@??f??@!??f??@*      ??!       2	?u?;O<???u?;O<??!?u?;O<??:	\;Q)@\;Q)@!\;Q)@B      ??!       J	LU?????LU?????!LU?????R      ??!       Z	LU?????LU?????!LU?????b      ??!       JGPUY?JTW??@b qӎ?GI@y???ؚF@?"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??Ǝ?H??!??Ǝ?H??0"g
;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??N:EC??!?O?+J???0"8
sequential/conv2d_1/Conv2DConv2D?Njb˰?!p{w??'??0"g
<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?[?L???!???G????0"i
=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterR???
??!4??iF^??0"i
=gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?????У?!k	?W???0"6
sequential/conv2d/Conv2DConv2D@???c???!??^;?N??0"[
:gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGrad???}@??!ߕN?2??":
sequential/conv2d_2/Relu_FusedConv2Dt?5GJ??!??`^
??"5
sequential/dense/MatMulMatMul??u???!??????0Q      Y@Y:??s?9'@ac?1?V@q?????4@y?«????"?
both?Your program is POTENTIALLY input-bound because 15.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?34.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?20.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 