import os
import shutil

#训练集和测试集划分
def readDirectory(folder,new_used_file,new_unused_file):
	if not os.path.exists(new_used_file):
		os.makedirs(new_used_file)
	if not os.path.exists(new_unused_file):
		os.makedirs(new_unused_file)

	folder_lists=os.listdir(folder)
	folder_lists=sorted(folder_lists)
	for folder_list in folder_lists:
		folder_list_new=folder+'/'+folder_list
		file_list=os.listdir(folder_list_new)

		for file in file_list[:-2]:
			src=os.path.join(folder_list_new,file)
			dst=os.path.join(new_used_file,file)
			shutil.copy(src,dst)
		src=os.path.join(folder_list_new,file_list[-1])
		dst=os.path.join(new_unused_file,file_list[-1])
		shutil.copy(src,dst)

readDirectory('./FaceV5','./used','unused')