# 20260928

## 202609281110

现在,我想在自己采集的数据集上测试reloc3r的效果。

数据录的我存储在/data/data/reLocationData/zc_data，相关传感器的参数见/data/data/location_data/zc_data/研发大楼/022419/calib/hesai_022419。用20260921_064638.mcap用来建图，其余两个包用于查询，给出模型最终重定位的精度。

环境使用/data/software/conda/anaconda3/envs/mv2d

先分析应该怎么做，然后进行测试，给出最终的测试结果。将结果放在docs/zc数据集上测试中，用文档保存下来，文档的结尾要加上时间戳。

你先给出实现方案，我评审后没有问题后，然后再进行测试。现在给出方案

## 202609281618

现在，再次在/data/data/reLocationData/zc_data/研发大楼数据这个数据中进行测试reloc3R的效果，将20260923_055425_real_robot_record_external_lz4来建图，20260923_060415_real_robot_record_external_lz4作为查询。

同时，给出整个算法耗时的gpu,cpu资源占用统计结果
