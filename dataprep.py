import os
import scipy.misc

root='D:/Parth Kashikar/AI/Projects/CAN/wikiart/wikiart/'
i = 0

path = []
title_heads = []

for subdirs,dirs,files in os.walk(root):
    if(i>0):
        path.append(subdirs)
        subdirs = subdirs.replace('D:/Parth Kashikar/AI/Projects/CAN/wikiart/wikiart/','')
        title_heads.append(subdirs)
    i=i+1


for ind in range(len(path)):
    if(ind>0):
        path_full = path[ind]
        save_path = ('D:/Parth Kashikar/AI/Projects/CAN/smallimages/' + title_heads[ind] + '/')
        os.mkdir(save_path)
        print("Now working on " + title_heads[ind])
        for subdirs,firs,files in os.walk(path_full):
            j = 0
            for f in files:
                if(j%500 ==0):
                    print(str(j) + ' files done.')
                image = scipy.misc.imread(path_full + '/' + f)
                image = scipy.misc.imresize(image,(256,256))
                name = f.replace('.jpg','.png')
                scipy.misc.imsave(save_path + name,image)
                j = j + 1


    