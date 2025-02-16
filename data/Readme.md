# Netease-Dense Dataset
Here, we provide the Netease-Dense Dataset. Due to the limit on size of the uploaded datatset, we uploaded it to the [Google Drive](https://drive.google.com/drive/folders/1h3IagvaD6hkwPf1y0-b1u4fbM6y7Wfuk?usp=sharing).

The data statistics and structure are shown below:
|#entity |#num.| 
| --- | --- | 
| user | 8,346 | 
| list | 10,871 |
| song | 100,584 | 
| user-list interactions | 838,640 |
| user-song interactions | 6,319,668 | 

- raw
  - user_list.txt
  - user_song.txt
  - list_song.txt
  - list_genre.txt

- processed
  - train (for list recommendation training)
  - test (for list recommendation evaluation)
  - train_item (training dataset for pretraining user-item interactions)
  - test_item (testing dataset for pretraining user-item interactions)
  -  _neighbor500.txt (the set containing 500 neighbors based on different meta-paths)
  -  _pretrain_samll.npy (pretrained embeddings for textual information)

