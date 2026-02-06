#!/bin/bash

## B4096
vai_c_xir -x models/YOLOv11_int.xmodel -a ./Architectures/arch_B4096.json -o zynq_output/yolov11n/ -n yolov11n

## B3136
#vai_c_xir -x models/YOLOv11_int.xmodel -a ./Architectures/arch_B3136.json -o zynq_output/yolov11n/ -n yolov8n

## B2304
#vai_c_xir -x models/YOLOv11_int.xmodel -a ./Architectures/arch_B2304.json -o zynq_output/yolov11n/ -n yolov11n

## B1600
#vai_c_xir -x models/YOLOv11_int.xmodel -a ./Architectures/arch_B1600.json -o zynq_output/yolov11n/ -n yolov11n

## B1152
#vai_c_xir -x models/YOLOv11_int.xmodel -a ./Architectures/arch_B1152.json -o zynq_output/yolov11n/ -n yolov11n

## B1024
#vai_c_xir -x models/YOLOv11_int.xmodel -a ./Architectures/arch_B1024.json -o zynq_output/yolov11n/ -n yolov11n

## B800
#vai_c_xir -x models/YOLOv11_int.xmodel -a ./Architectures/arch_B800.json -o zynq_output/yolov11n/ -n yolov11n

## B512
#vai_c_xir -x models/YOLOv11_int.xmodel -a ./Architectures/arch_B512.json -o zynq_output/yolov11n/ -n yolov11n
