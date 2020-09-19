import os
import re
import sys
import cv2
import time
import datetime
import shutil
import numpy as np
import pandas as pd
import pydicom as dicom
from skimage import feature
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

ROOT_PATH = './database'
LB_PATH = ROOT_PATH + '/lung_blocks'
NM_PATH = ROOT_PATH + '/normalized'
EXT_PATH = './extraction'

def scanDatabase(rootpath):
    print(f"scanDatabase({rootpath})")
    print('This may take a lot of time')
    start_time = time.time()

    fileList = {'img_id': [],
                'img_name': [],
                'roi_number': [],
                'coord_x': [],
                'coord_y': [],
                'percent': [],
                'label': [],
                'patient_id': [],
                'path': [],
                'intercept': [],
                'slope': [],
                'minHU': [],
                'maxHU': [] }
    ids = 0
    for dirpath, dirnames, filenames in os.walk(rootpath,topdown=True):
        unwanted = [] #same patient, another device
        for dir in dirnames:
            #directories that doesn't match with pattern of patient_id
            if not re.match('\d{3}|\d{2}|\d',dir) and dir != 'HRCT_pilot':
                unwanted.append(dir)

        if len(unwanted) > 0:
            unwanted.sort()
            dirnames[:] = [unwanted[0]]

        for file in filenames:
            if '.dcm' in file:
                ids += 1
                fileList['img_id'].append(ids)

                #get infos from image name
                infos = file.replace('.dcm','').split('_')
                fileList['img_name'].append(infos[0])
                fileList['roi_number'].append(infos[1])

                x,y = infos[2].split('-')
                fileList['coord_x'].append(x)
                fileList['coord_y'].append(y)

                fileList['percent'].append(infos[3])
                fileList['label'].append(int(infos[4]))

                #get infos from pathname
                fileList['path'].append( dirpath.replace(ROOT_PATH,'') + '/' + file)
                fileList['patient_id'].append( next(filter(lambda x: re.match('\d{3}|\d{2}|\d',x),dirpath.split('/'))) )

                with dicom.dcmread( dirpath + '/' + file ) as dicom_img:
                    fileList['intercept'].append(dicom_img.RescaleIntercept)
                    fileList['slope'].append(dicom_img.RescaleSlope)
                    fileList['minHU'].append(np.min(dicom_img.pixel_array))
                    fileList['maxHU'].append(np.max(dicom_img.pixel_array))


    df = pd.DataFrame.from_dict(fileList)
    print(
        f"finish with {len(df.loc[df['label'] != 6])} datas found - Time elapsed: {datetime.timedelta(seconds=round(time.time()-start_time))} seconds")
    return df.loc[df['label'] != 6]


def getLungBlocks(force=False):
    if not os.path.isdir(LB_PATH):
        print(f"{LB_PATH} not found")
        sys.exit()
    elif os.path.isfile(f"{LB_PATH}.csv") and not force:
        lung_blocks_df = pd.read_csv(f"{LB_PATH}.csv")
        print(f"{LB_PATH}.csv loaded")
    else:
        lung_blocks_df = scanDatabase(LB_PATH)
        lung_blocks_df.to_csv(f"{LB_PATH}.csv",index=False)
        print(f"{LB_PATH}.csv saved")

    return lung_blocks_df


def normalizeHU(img, slope, intercept, min, max):
    img = ( img - intercept ) / slope
    img = (img - min) / ((max-min) / 255)
    img = np.rint(img)
    return img

def getNormalizedImages(lung_blocks_df=None, force=False):
    if os.path.isfile(f"{NM_PATH}.csv") and not force:
        normalized_df = pd.read_csv(f"{NM_PATH}.csv")
        print(f"{NM_PATH}.csv loaded")

    elif isinstance(lung_blocks_df,pd.DataFrame):
        print('normalizing')
        start_time = time.time()

        if os.path.isdir(NM_PATH):
            shutil.rmtree(NM_PATH)

        os.mkdir(NM_PATH)

        minHU = np.min(lung_blocks_df['minHU'].to_numpy())
        maxHU = np.max(lung_blocks_df['maxHU'].to_numpy())

        ids = lung_blocks_df['img_id'].to_numpy()
        paths = lung_blocks_df['path'].to_numpy()
        slopes = lung_blocks_df['slope'].to_numpy()
        intercepts = lung_blocks_df['intercept'].to_numpy()

        normalized_path = np.empty(len(ids),dtype='U64')

        for i in range(len(ids)):
            with dicom.dcmread( ROOT_PATH + paths[i] ) as dicom_img:
                normalized_img = normalizeHU(dicom_img.pixel_array, slopes[i], intercepts[i], minHU, maxHU)
                img_path = f"{NM_PATH}/{ids[i]}.png"
                plt.imsave(img_path, normalized_img,cmap='gray')
                normalized_path[i] = img_path.replace(ROOT_PATH,'')

        print('{} images normalized - Time elapsed: {} seconds'.format(len(ids),datetime.timedelta(seconds=round(time.time()-start_time))))

        normalized_df = lung_blocks_df[['img_id','patient_id','label']].copy()
        normalized_df['path'] = normalized_path
        normalized_df.to_csv(f'{NM_PATH}.csv',index=False)

        print(f"{NM_PATH}.csv saved")

    else:
        print('Need lung_blocks_df to normalize')
        lung_blocks_df = getLungBlocks()
        return getNormalizedImages(lung_blocks_df,force=force)

    return normalized_df


def getLBP(normalized_imgs_df=None, R=1, method='default', force=False):
    if os.path.isfile(EXT_PATH + '/lbp.npz') and not force:
        print(f"{EXT_PATH}/lbp.npz loaded")
        dataset = np.load(EXT_PATH + '/lbp.npz',allow_pickle=True)['dataset']
    elif not isinstance(normalized_imgs_df,pd.DataFrame):
        print('Need normalized_imgs_df to extract')
        normalized_imgs_df = getNormalizedImages()
        return getLBP(normalized_imgs_df,force=force)
    else:
        print('extracting lbp')
        start_time = time.time()

        P = R*8 ## number of neighbours

        patients = normalized_imgs_df['patient_id'].unique()
        dataset = np.ndarray(len(patients),dtype='object')

        for patient,index in zip(patients,range(len(patients))):
            patient_df = normalized_imgs_df.loc[normalized_imgs_df['patient_id'] == patient]
            paths = patient_df['path'].to_numpy()
            data = []
            for i in range(len(paths)):
                image = cv2.imread( ROOT_PATH + paths[i], cv2.IMREAD_GRAYSCALE)
                lbp_img = feature.local_binary_pattern(image,P,R,method).astype(int).ravel()
                lbp_hist = np.zeros(256,dtype=np.uint16)
                for pixel in lbp_img:
                    lbp_hist[pixel]+=1
                data.append(lbp_hist)

            target = patient_df['label'].values.tolist()
            dataset[index] = {'patient_id': patient,
                              'data': data, 'target': target}

        print(f"{len(normalized_imgs_df)} lbp extracted - Time elapsed: {datetime.timedelta(seconds=round(time.time()-start_time))} seconds")

        if( not os.path.isdir(EXT_PATH) ):
            os.mkdir(EXT_PATH)

        with open( EXT_PATH + '/lbp.npz', 'wb') as out:
            np.savez(out,dataset=dataset)
            print(f"{EXT_PATH}/lbp.npz saved")

    #to list because we wanna use splits (like [:index] + [index+1:]) and numpy is dumb
    return dataset.tolist()


def getTopHat(normalized_imgs_df=None, filterSize=(3, 3), method='default', force=False):
    if os.path.isfile(EXT_PATH + '/tophat.npz') and not force:
        print(f"{EXT_PATH}/tophat.npz loaded")
        dataset = np.load(EXT_PATH + '/tophat.npz',
                          allow_pickle=True)['dataset']
    elif not isinstance(normalized_imgs_df, pd.DataFrame):
        print('Need normalized_imgs_df to extract')
        normalized_imgs_df = getNormalizedImages()
        return getTopHat(normalized_imgs_df, force=force)
    else:
        print('extracting tophat')
        start_time = time.time()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, filterSize)

        patients = normalized_imgs_df['patient_id'].unique()
        dataset = np.ndarray(len(patients), dtype='object')

        for patient, index in zip(patients, range(len(patients))):
            patient_df = normalized_imgs_df.loc[normalized_imgs_df['patient_id'] == patient]
            paths = patient_df['path'].to_numpy()
            data = []
            for i in range(len(paths)):
                image = cv2.imread(ROOT_PATH + paths[i], cv2.IMREAD_GRAYSCALE)
                # TOPHAT
                # (diferenca entre a imagem original e a abertura dela)
                tophat_img = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

                # BLACKHAT
                # (diferenca entre a imagem original e o fechamento dela)
                blackhat_img = cv2.morphologyEx(
                    image, cv2.MORPH_BLACKHAT, kernel)

                mean_tophat, std_tophat = cv2.meanStdDev(tophat_img)
                mean_blackhat, std_blackhat = cv2.meanStdDev(blackhat_img)

                data.append([mean_tophat.item(0), std_tophat.item(0),
                             mean_blackhat.item(0), std_blackhat.item(0)])

            target = patient_df['label'].values.tolist()
            dataset[index] = {'patient_id': patient,
                              'data': data, 'target': target}

        print(f"{len(normalized_imgs_df)} tophat extracted - Time elapsed: {datetime.timedelta(seconds=round(time.time()-start_time))} seconds")

        if(not os.path.isdir(EXT_PATH)):
            os.mkdir(EXT_PATH)

        with open(EXT_PATH + '/tophat.npz', 'wb') as out:
            np.savez(out, dataset=dataset)
            print(f"{EXT_PATH}/tophat.npz saved")

    #to list because we wanna use splits (like [:index] + [index+1:]) and numpy is dumb
    return dataset.tolist()


def main():
    # lb_df = getLungBlocks(force=False)
    # nimg_df = getNormalizedImages(lb_df, force=False)
    getLBP(nimg_df, force=False)
    getTopHat(nimg_df, force=False)


if __name__ == "__main__":
    main()
