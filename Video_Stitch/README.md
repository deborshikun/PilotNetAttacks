1. Download ffmpeg (winget install ffmpeg)
2. Command on terminal:

    ffmpeg -framerate 24 -start_number 10550 -i "path to images" -c:v libvpx-vp9 -crf 31 -b:v 0 "output path/output file name.webm"

# ffmpeg -framerate 24 -start_number 10550 -i "C:\Users\beybl\Desktop\PilotNetAttacks\Using_torchattacks\MIFGSM\verification_adv_img_eps003_alpha0007_steps10_decay1\verification_%05d.jpg" -c:v libvpx-vp9 -crf 31 -b:v 0 "C:\Users\beybl\Desktop\PilotNetAttacks\Video_Stitch\MIFGSM\Verification_003_0007_10_1_MIFGSM.webm"

A few points on this one:

-r 24: This sets the frames per second.
-i path/to/frame_%04d.png: The path to the png images to stitch. %04d refers to 4 successive digits in a row that count up from 1 - e.g. 0001, 0002, etc. %d would mean 1, 2, ...., and %02d for example would mean 01, 02, 03....
-c:v libvpx-vp9: The codec. We're exporting to webm, and vp9 is the latest available codec for webm video files.
-b:v 2000k: The bitrate. Higher values mean more bits will be used per second of video to encode detail. Raise to increase quality.
-crf 31: The encoding quality. Should be anumber betwee 0 and 63 - higher means lower quality and a lower filesize.