#!/bin/bash
mkdir /project/3dlg-hcvc/motionnet/extract_logs
scp hanxiao@cedar.computecanada.ca:/scratch/hanxiao/proj-motionnet/extract.zip /project/3dlg-hcvc/motionnet
unzip /project/3dlg-hcvc/motionnet/extract.zip -d /project/3dlg-hcvc/motionnet/extract_logs/
rm -rf /project/3dlg-hcvc/motionnet/extract.zip