1. Generate file lists:
	```
		python generate_file_list.py
	```
2. Download VGG-19 weight file from https://github.com/machrisaa/tensorflow-vgg
3. Train:
	```
		python train_MBLLVEN_raw.py
	```
4. Test:
	```
		python test_MBLLVEN.py
	```
5. Get quantitative measurements:
	```python psnr_ssim_mabd.py```