#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import sys
# In[2]:
def CemalGuven_Adal_21703986_hw2(question):
    if question == '1':
        # ---------------------------------------------------------------------------Question1--------------------------------------------------------------------
        # Question 1 A------------------------------------------
        # Defining array A which is given in the Homework question 1
        # In[2]:

        # loading .mat file to python
        data1 = sio.loadmat('c2p3.mat')
        data1.keys()

        

        # breaking down stim and counts from.mat file and checking their sizes
        stim = data1['stim']
        counts = data1['counts']
        print(np.shape(stim))
        print(np.shape(counts))

       

   

        # STA function
        def STA(counts, stim):
            average = np.zeros((16, 16, 10))
            for i in range(10, counts.shape[0]):
                average = average + stim[:, :, i - 10:i] * counts[i, :]
            average = average / (np.sum(counts))

            return average

        

        aver = STA(counts, stim)
        print(np.size(aver))
        for i in range(0, 10):
            plt.figure()
            plt.imshow(aver[:, :, i], vmin=np.min(aver), vmax=np.max(aver), cmap='gray')
            plt.colorbar()
            plt.title(str(10 - i) + 'Steps Before Spike')

   

        # partb
        plt.figure()
        averagerow = np.sum(aver, axis=0)
        plt.title("Row Summed Averages")
        plt.imshow(averagerow, cmap='gray')

        plt.figure()
        averagecol = np.sum(aver, axis=1)
        plt.title("Coloumn Summed Averages")
        plt.imshow(averagerow, cmap='gray')

        

        # PARTC
        # for all stimlus
        aver1 = aver[:, :, -1]  # for takng the last one
        hist1 = np.zeros(32767)
        for i in range(32767):
            hist1[i] = np.sum(aver1 * stim[:, :, i])
        plt.figure(figsize=(8, 10))
        plt.hist(hist1, density='True', bins=90, rwidth=0.70)
        plt.title('Stimulus Projection for all Spikes')
        plt.show()
        # for stimulus non 0's
        hist2 = np.array([])

        for i in range(32767):
            if counts[i, :] != 0:
                sumvalue = np.sum(aver1 * stim[:, :, i])
                hist2 = np.append(hist2, sumvalue)
        plt.figure(figsize=(8, 10))
        plt.hist(hist2, density='True', bins=90, rwidth=0.70)
        plt.title('Stimulus Projection for non zero Spikes')
        plt.show()

       


    elif question == '2':
        # part a
        def Calcul(x, y):
            thetaS = 4
            thetaC = 2
            a = (1 / (2 * np.pi * thetaC * thetaC)) * np.exp((-(x * x + y * y) / (2 * thetaC * thetaC))) - (
                        1 / (2 * np.pi * thetaS * thetaS)) * np.exp((-(x * x + y * y) / (2 * thetaS * thetaS)))
            return a

        Dog = np.zeros((21, 21))
        for x in range(-10, 10):
            for y in range(-10, 10):
                a = Calcul(x, y)
                Dog[x + 10, y + 10] = a
        plt.figure()
        plt.imshow(Dog)
        plt.colorbar()
        plt.title('DOG Receptive Field')
        np.shape(Dog)

       

        # part b
        monk = plt.imread("hw2_image.bmp")
        plt.figure()
        plt.imshow(monk)
        plt.title('original image')
        np.shape(monk)
        monk2 = np.zeros((480, 512))
        print(np.shape(monk))
        for i in range(0, 480):
            for k in range(0, 512):
                i1 = i
                k1 = k
                if i1 <= 10:
                    i1 = 10
                if i1 >= 470:
                    i1 = 469
                if k1 <= 10:
                    k1 = 10
                if k1 >= 502:
                    k1 = 501
                h = monk[i1 - 10:i1 + 11, k1 - 10:k1 + 11, 0] * Dog
                monk2[i, k] = np.sum(h)
        plt.figure()
        plt.imshow(monk2, cmap='gray')
        plt.title('Neural Activity')
        plt.colorbar()

    

        # partc
        def edgedetec(h, threshold):
            a = np.zeros((480, 512))

            for i in range(0, 480):
                for k in range(0, 512):
                    if h[i, k] < threshold:
                        a[i, k] = 0
                    else:
                        a[i, k] = 1
            return a

        edgedetected = edgedetec(monk2, 2)
        plt.figure()
        plt.imshow(edgedetected, cmap='gray')
        plt.title("threshhold=2")
        edgedetected = edgedetec(monk2, 10)
        plt.figure()
        plt.imshow(edgedetected, cmap='gray')
        plt.title("threshhold=10")
        edgedetected = edgedetec(monk2, 1)
        plt.figure()
        plt.imshow(edgedetected, cmap='gray')
        plt.title("threshhold=1")
        edgedetected = edgedetec(monk2, 0)
        plt.figure()
        plt.imshow(edgedetected, cmap='gray')
        plt.title("threshhold=0")
        edgedetected = edgedetec(monk2, -1)
        plt.figure()
        plt.imshow(edgedetected, cmap='gray')
        plt.title("threshhold= -1")
        edgedetected = edgedetec(monk2, -2)
        plt.figure()
        plt.imshow(edgedetected, cmap='gray')
        plt.title("threshhold=-2")

        

        # part d
        def Gabor(x, y, o):
            sigmaL = 3
            sigmaw = 3
            theta1 = 0
            landa = 6
            k = np.array([np.cos(o), np.sin(o)])
            kOrt = np.array([-np.sin(o), np.cos(o)])
            c = [x, y]
            var1 = np.dot(k, c) * np.dot(k, c) / (2 * sigmaL * sigmaL)
            var2 = np.dot(kOrt, c) * np.dot(kOrt, c) / (2 * sigmaw * sigmaw)
            var3 = np.cos(((2 * np.pi * (np.dot(kOrt, c))) / landa) + theta1)
            b = np.exp(-var1 - var2) * var3
            return b

        res = np.zeros((21, 21))
        for x in range(-10, 10):
            for y in range(-10, 10):
                b = Gabor(x, y, np.pi / 2)
                res[x + 10, y + 10] = b
        plt.figure()
        plt.imshow(res)
        plt.title('Receptive Gabor Field')
        plt.colorbar()
        np.shape(res)

        

        # part e
        monke = plt.imread("hw2_image.bmp")
        plt.figure()
        plt.imshow(monke)
        plt.title('original image')
        np.shape(monke)
        monk3 = np.zeros((480, 512))
        print(np.shape(monke))
        for i in range(0, 480):
            for k in range(0, 512):
                i1 = i
                k1 = k
                if i1 <= 10:
                    i1 = 10
                if i1 >= 470:
                    i1 = 469
                if k1 <= 10:
                    k1 = 10
                if k1 >= 502:
                    k1 = 501
                h1 = monke[i1 - 10:i1 + 11, k1 - 10:k1 + 11, 0] * res
                monk3[i, k] = np.sum(h1)
        plt.figure()
        plt.imshow(monk3, cmap='gray')
        plt.title('Gabor Filtred')
        plt.colorbar()

        

        # part f
        # 0
        resf1 = np.zeros((21, 21))
        for x in range(-10, 10):
            for y in range(-10, 10):
                b = Gabor(x, y, 0)
                resf1[x + 10, y + 10] = b
        plt.figure()
        plt.title('Gabor Receptive Field theta=0')
        plt.imshow(resf1)
        monkf1 = np.zeros((480, 512))
        for i in range(0, 480):
            for k in range(0, 512):
                i1 = i
                k1 = k
                if i1 <= 10:
                    i1 = 10
                if i1 >= 470:
                    i1 = 469
                if k1 <= 10:
                    k1 = 10
                if k1 >= 502:
                    k1 = 501
                hf1 = monke[i1 - 10:i1 + 11, k1 - 10:k1 + 11, 0] * resf1
                monkf1[i, k] = np.sum(hf1)
        plt.figure()
        plt.imshow(monkf1, cmap='gray')
        plt.title('Gabor Filtered Image theta=0')
        plt.colorbar()
        # pi/6

        resf2 = np.zeros((21, 21))
        for x in range(-10, 10):
            for y in range(-10, 10):
                b = Gabor(x, y, np.pi / 6)
                resf2[x + 10, y + 10] = b
        plt.figure()
        plt.imshow(resf2)
        plt.title('Gabor Receptive Field theta=pi/6')

        monkf2 = np.zeros((480, 512))
        for i in range(0, 480):
            for k in range(0, 512):
                i1 = i
                k1 = k
                if i1 <= 10:
                    i1 = 10
                if i1 >= 470:
                    i1 = 469
                if k1 <= 10:
                    k1 = 10
                if k1 >= 502:
                    k1 = 501
                hf2 = monke[i1 - 10:i1 + 11, k1 - 10:k1 + 11, 0] * resf2
                monkf2[i, k] = np.sum(hf2)
        plt.figure()
        plt.imshow(monkf2, cmap='gray')
        plt.title('Gabor Filtered Image theta=pi/6')

        plt.colorbar()

        # pi/3

        resf3 = np.zeros((21, 21))
        for x in range(-10, 10):
            for y in range(-10, 10):
                b = Gabor(x, y, np.pi / 3)
                resf3[x + 10, y + 10] = b
        plt.figure()
        plt.imshow(resf3)
        plt.title('Gabor Receptive Field theta=pi/3')

        monkf3 = np.zeros((480, 512))
        for i in range(0, 480):
            for k in range(0, 512):
                i1 = i
                k1 = k
                if i1 <= 10:
                    i1 = 10
                if i1 >= 470:
                    i1 = 469
                if k1 <= 10:
                    k1 = 10
                if k1 >= 502:
                    k1 = 501
                hf3 = monke[i1 - 10:i1 + 11, k1 - 10:k1 + 11, 0] * resf3
                monkf3[i, k] = np.sum(hf3)
        plt.figure()
        plt.imshow(monkf3, cmap='gray')
        plt.title('Gabor Filtered Image theta=pi/3')

        plt.colorbar()

        # pi/2

        resf4 = np.zeros((21, 21))
        for x in range(-10, 10):
            for y in range(-10, 10):
                b = Gabor(x, y, np.pi / 2)
                resf4[x + 10, y + 10] = b
        plt.figure()
        plt.imshow(resf4)
        plt.title('Gabor Receptive Field theta=pi/2')

        monkf4 = np.zeros((480, 512))
        for i in range(0, 480):
            for k in range(0, 512):
                i1 = i
                k1 = k
                if i1 <= 10:
                    i1 = 10
                if i1 >= 470:
                    i1 = 469
                if k1 <= 10:
                    k1 = 10
                if k1 >= 502:
                    k1 = 501
                hf4 = monke[i1 - 10:i1 + 11, k1 - 10:k1 + 11, 0] * resf4
                monkf4[i, k] = np.sum(hf4)
        plt.figure()
        plt.imshow(monkf4, cmap='gray')
        plt.title('Gabor Filtered Image theta=pi/2')
        plt.colorbar()
        # sum of all
        allmonkey = monkf1 + monkf2 + monkf3 + monkf4
        plt.figure()
        plt.imshow(allmonkey, cmap='gray')
        plt.title('Gabor Filtered Sum')
        plt.colorbar()

       

        edgedetectedmonkey = edgedetec(allmonkey, 0)
        plt.figure()
        plt.imshow(edgedetectedmonkey, cmap='gray')
        plt.title('Threshold=0')
        edgedetectedmonkey = edgedetec(allmonkey, 400)
        plt.figure()
        plt.imshow(edgedetectedmonkey, cmap='gray')
        plt.title('Threshold=400')
        edgedetectedmonkey = edgedetec(allmonkey, 1000)
        plt.figure()
        plt.imshow(edgedetectedmonkey, cmap='gray')
        plt.title('Threshold=1000')
        edgedetectedmonkey = edgedetec(allmonkey, -10)
        plt.figure()
        plt.imshow(edgedetectedmonkey, cmap='gray')
        plt.title('Threshold=-10')
        edgedetectedmonkey = edgedetec(allmonkey, 100)
        plt.figure()
        plt.imshow(edgedetectedmonkey, cmap='gray')
        plt.title('Threshold=100')

        # In[ ]:

        # -------------------QUESION2--------------------------------------------------------------------------------------------------




question = input("enter question number")
CemalGuven_Adal_21703986_hw2(question)


# In[ ]:





# In[ ]:




