?	?A?<?I7@?A?<?I7@!?A?<?I7@	ȒCp????ȒCp????!ȒCp????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?A?<?I7@!\?zz??1?:?? B5@A????(??IM?]~??Y?Oqx???rEagerKernelExecute 0*	??|?5b@2F
Iterator::Model????}ɲ?!?y0,?rI@);ŪA???1??]O?B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatZ?xZ~???!
&f???9@)^??yȔ??1????wc5@:Preprocessing2U
Iterator::Model::ParallelMapV2??C?r???!%?9?+@)??C?r???1%?9?+@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?IbI????!?Gwg?3@)4?Op???1???Q)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice+?򑔄?!l??`W?@)+?򑔄?1l??`W?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorZ???аx?!|?сϸ@)Z???аx?1|?сϸ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipk?SUh ??!]???|?H@)`=?[?w?1???B@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ȒCp????I`?f? @Q??V?l?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!\?zz??!\?zz??!!\?zz??      ??!       "	?:?? B5@?:?? B5@!?:?? B5@*      ??!       2	????(??????(??!????(??:	M?]~??M?]~??!M?]~??B      ??!       J	?Oqx????Oqx???!?Oqx???R      ??!       Z	?Oqx????Oqx???!?Oqx???b      ??!       JGPUYȒCp????b q`?f? @y??V?l?V@?"5
sequential/dense/MatMulMatMul??L???!??L???0"b
7sequential/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor?>??kR??!?n??????"x
Igradient_tape/sequential/up_sampling2d_1/resize/ResizeNearestNeighborGradResizeNearestNeighborGrad??g??!???e????"i
=gradient_tape/sequential/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterr-"IP??!`?É????0"i
=gradient_tape/sequential/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter*??i???!????????0"K
$Adam/Adam/update_8/ResourceApplyAdamResourceApplyAdam?ĨO1ߤ?!????y??"C
'gradient_tape/sequential/dense/MatMul_1MatMul?FU?S???!?SgY????"C
%gradient_tape/sequential/dense/MatMulMatMul?R&????!?q???0"I
0gradient_tape/sequential/dropout_1/dropout/Mul_2Mul???????!?h?؎??";
"sequential/dropout_1/dropout/Mul_1Mul?>??????!?B$n???Q      Y@YAd?W?,)@ax??g?U@q??#???0@y?r4m?5??"?

both?Your program is POTENTIALLY input-bound because 6.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?16.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Pascal)(: B 