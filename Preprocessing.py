import model_torch
import modules_torch
import _utils_torch
from _utils_torch import *
import loss

def preprocessing(train_data_dir, test_data_dir):

    train_list = os.listdir(train_data_dir)
    test_list = os.listdir(test_data_dir)
    
    X_train = []
    Y_train = []
    for item in tqdm(train_list):
        file_names = train_data_dir + '/' + item + '/*msk.nii.gz'
        sam_mask = glob.glob(file_names)
        sam_img = file_names.replace('derivatives', 'rawdata')
        sam_img = sam_img.replace('msk', 'ct')
        sam_img = glob.glob(sam_img)
    
        sam_img = sam_img[0]
        sam_mask = sam_mask[0]
    
        sam_img = sitk.ReadImage(sam_img)
        sam_mask = sitk.ReadImage(sam_mask)
    
        assert sam_img.GetSize() == sam_mask.GetSize()
    
        sam_img = sitk.GetArrayFromImage(sam_img)
        sam_mask = sitk.GetArrayFromImage(sam_mask)
        if sam_img.shape[1:] != (512, 512):
            continue
        if sam_mask.max() == 28:
            continue
        sam_img = normalize_dcm(sam_img)
    
        sam_img = np.transpose(sam_img, axes=(2,0,1))
        sam_img = sam_img[:,::-1,:]
        sam_mask = np.transpose(sam_mask, axes=(2,0,1))
        sam_mask = sam_mask[:,::-1,:]
        
        print(item)
        print(sam_img.shape, sam_img.min(), sam_img.max(), sam_img.dtype, '\n')
        
        for _slice in range(len(sam_img)):
            img_slice = resize(sam_img[_slice,...],(512,512),anti_aliasing=True,preserve_range=True,order=0)
            mask_slice = np.zeros((512,512))
            for idx in range(int(sam_mask[_slice,...].max())):
                value = int(idx+1)
                dummy = np.zeros(sam_mask[_slice,...].shape)
                dummy[sam_mask[_slice,...]==value] = 1
                dummy = resize(dummy,(512,512),anti_aliasing=False,preserve_range=True,order=0)
                mask_slice[dummy==1] = value
            X_train.append(img_slice)
            Y_train.append(mask_slice)
    
    X_test = []
    Y_test = []
    for item in tqdm(test_list):
        file_names = test_data_dir + '/' + item + '/*msk.nii.gz'
        sam_mask = glob.glob(file_names)
        sam_img = file_names.replace('derivatives', 'rawdata')
        sam_img = sam_img.replace('msk', 'ct')
        sam_img = glob.glob(sam_img)
    
        sam_img = sam_img[0]
        sam_mask = sam_mask[0]
    
        sam_img = sitk.ReadImage(sam_img)
        sam_mask = sitk.ReadImage(sam_mask)
    
        assert sam_img.GetSize() == sam_mask.GetSize()
    
        sam_img = sitk.GetArrayFromImage(sam_img)
        sam_mask = sitk.GetArrayFromImage(sam_mask)
        if sam_img.shape[1:] != (512, 512):
            continue
        if sam_mask.max() == 28:
            continue
        sam_img = normalize_dcm(sam_img)
    
        sam_img = np.transpose(sam_img, axes=(2,0,1))
        sam_img = sam_img[:,::-1,:]
        sam_mask = np.transpose(sam_mask, axes=(2,0,1))
        sam_mask = sam_mask[:,::-1,:]
    
        print(item)
        print(sam_img.shape, sam_img.min(), sam_img.max(), sam_img.dtype, '\n')
        
        for _slice in range(len(sam_img)):
            img_slice = resize(sam_img[_slice,...],(512,512),anti_aliasing=True,preserve_range=True,order=0)
            mask_slice = np.zeros((512,512))
            for idx in range(int(sam_mask[_slice,...].max())):
                value = int(idx+1)
                dummy = np.zeros(sam_mask[_slice,...].shape)
                dummy[sam_mask[_slice,...]==value] = 1
                dummy = resize(dummy,(512,512),anti_aliasing=False,preserve_range=True,order=0)
                mask_slice[dummy==1] = value
            X_test.append(img_slice)
            Y_test.append(mask_slice)
            
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test