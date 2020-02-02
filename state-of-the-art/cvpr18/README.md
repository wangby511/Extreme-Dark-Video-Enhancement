1. Change dataset from video pairs to image pairs:
	```python prepare_data_for_cvpr18_network.py```
2. Train:
	```python train_downsampled_he2he.py```
3. Test:
	```python test_testset_downsampled_he2he.py```
4. Compile test results from images to videos:
	```
		cd result_downsampled_he2he
		python organize_test_results.py
		cd ..
	``` 
5. Get quantitative measurements:
	```python psnr_ssim_mabd.py```