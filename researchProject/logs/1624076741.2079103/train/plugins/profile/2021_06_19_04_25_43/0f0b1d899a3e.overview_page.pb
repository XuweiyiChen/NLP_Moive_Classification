?	???0?? @???0?? @!???0?? @	af%??v.@af%??v.@!af%??v.@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL???0?? @?>+??1tѐ?(?@AE?4f??Ivnڌ??@YJ??1??rEagerKernelExecute 0*	??Q?Z?@2U
Iterator::Model::ParallelMapV2Ɔn?J??!???j?R@)Ɔn?J??1???j?R@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice}?R??c??!?gOu??-@)}?R??c??1?gOu??-@:Preprocessing2F
Iterator::Model?3??????!??oW|?S@)%Ί??>??1<?ڔ?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?U1?~??!??kԁ?@)?[?~l???1^c?9 @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap,H3Mg??!t^?ӻ?0@){????1U?Ǵ
 @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?9}=_??!??@??4@)?V?????1? ??R???:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??????u?!?t??<???)??????u?1?t??<???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 15.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?29.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t18.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9af%??v.@I?q?H@Qm4?3??A@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?>+???>+??!?>+??      ??!       "	tѐ?(?@tѐ?(?@!tѐ?(?@*      ??!       2	E?4f??E?4f??!E?4f??:	vnڌ??@vnڌ??@!vnڌ??@B      ??!       J	J??1??J??1??!J??1??R      ??!       Z	J??1??J??1??!J??1??b      ??!       JGPUYaf%??v.@b q?q?H@ym4?3??A@?"i
=gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?GS.??!?GS.??0"g
;gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter~d?@E??!??t?????0"8
sequential/conv2d_1/Conv2DConv2D}???7???!????g??0"g
<gradient_tape/sequential/conv2d_1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?'f?Ԭ?!ũ|v???0"i
=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterD???,??!N<??A??0"i
=gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter/??X???!4֭?s???0"6
sequential/conv2d/Conv2DConv2D??p?ȣ?!???s7??0"[
:gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGradMaxPoolGradϽ?Ϥ#??!?՗J???":
sequential/conv2d_2/Relu_FusedConv2D??Iw??!>?5?????"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits2J?C???!????????Q      Y@Y@n]?G*@a8R4??U@q?D?t&@yp??½?"?
both?Your program is MODERATELY input-bound because 15.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?29.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.high"t18.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?11.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 