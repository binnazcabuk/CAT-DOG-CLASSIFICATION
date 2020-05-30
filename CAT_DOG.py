
# -*- coding: utf-8 -*-
"""


@author:b_c
"""

"""Not: 
Verilerimizin içinde bulunduğu klasoru working directory olarak seçmemiz gerekli.
Bunun için ilgili klasoru File explorer da açtıktan sonra ekranının sol 
kısmında yer alan ikona(options) tıklayarak "Select console workng directory"
seçmeniz yeterli. """
 


# Öncelikle gerekli kütüphane ve paketleri  indir
from keras.models import Sequential
# networku başlatmak için kullanılır. 
#CNN sequentil layerlara sahip olduğu için bunu kullanıyoruz.

from keras.layers import Conv2D
# resimler Conv2D olduğundan, video için Conv3D

from keras.layers import MaxPooling2D

from keras.layers import Flatten
#feature maps(özellik haritalarının) in tek bir büyük özellik vektörü boyut 
#altında toplanması amacıyla kullanılır.

from keras.layers import Dropout

from keras.layers import Dense
#fully connected layer oluşturmak için kullanılır.


# KISIM 1 CNN Oluştur

# CNN i başlat Başlat 
sınıflandırıcı= Sequential ()

""" Adım 1 Convolutional Layer
Bilgi almak için örneğin Conv2D üzerine farenizin imlecini yerleştirin.
ctrl + i ye bastığınızda ilgili özelliğe ait bilgiler ayrı bir ekranda 
listelenir."""

sınıflandırıcı.add(Conv2D(32,(3,3), input_shape= (64,64,3), activation='relu'))
""" 32 filtre sayısı
(3,3) filtre boyutu
 input_shape resimlerin boyutu ve channel sayısı RGB=3 BW=1
 training ve test de kullanılan tüm resimler aynı büyüklükte olmalı. Bu 
uygulamada tüm resimleri 64*64 e zorlayacağız.
 relu aktivasyon fonksiyonu """

# Adım 2 Pooling
# özellik haritalarının boyutunu küçültür.
sınıflandırıcı.add(MaxPooling2D (pool_size= (2,2)))

# İkinci CNN katmanının eklenmesi
# İlk layerdan farkı input_shape in belirtilmesine gerek yok.
sınıflandırıcı.add(Conv2D(32,(3,3), activation='relu'))
sınıflandırıcı.add(MaxPooling2D (pool_size= (2,2)))

sınıflandırıcı.add(Conv2D(64,(3,3), activation='relu'))
sınıflandırıcı.add(MaxPooling2D (pool_size= (2,2)))



# Adım 3 Flattening
# 1 boyutlu vektöre dönüştür.
sınıflandırıcı.add(Flatten())

# Adım 4 Full Connection
sınıflandırıcı.add(Dense(units= 64, activation='relu'))

sınıflandırıcı.add(Dropout(0.5))

sınıflandırıcı.add(Dense(units=1, activation='sigmoid'))

# CNN i Derle
sınıflandırıcı.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])
""" Kayıp(loss) fonksiyonu olarak binary-crossentropy kullandık. 
Çünkü iki sınıfımız var. Kedi ve Köpek  """

# KISIM 2 Resimleri CNN e Fit Etme

# Augmentation uyguluyoruz.  rescale önemli.
#Bu kısmı Keras Documentation--> Image Preprocessing sayfasından uyarladık.

# ImageDataGenerator u import etmeyi unutma!
from keras.preprocessing.image import ImageDataGenerator

# Overfitting i önlemek veri setimizi büyütmek için uyguluyoruz.
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# test verisinde sadece rescale uyguluyoruz. Tüm degerler 0-1 arasına çekilir.
test_datagen = ImageDataGenerator(rescale=1./255)

# altklasörden resimleri okur. Eğitim veri setimizi oluşturuyoruz.
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        # hedef dizin
        target_size=(64, 64),
        # tüm resimler 64*64 boyutuna dönüştürülecek
        batch_size=32,
        # toplu iş boyutu. Her Gradient update de kullanılacak örnek sayısı
        class_mode='binary'
        #binary_crossentropy kullandığımız için binary labels(etiketlere) 
        #ihtiyacımız var.
        )
#Test veri setimizi oluşturuyoruz.
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

""" Ağı eğitip test ediyoruz. Aşağıdaki kod 50 epoch a sahip. 
Çalıştırdığınız zaman sonucu elde etmeniz zaman alabilir. daha düşük epoch sayılarını deneyebilirsiniz.
"""
sınıflandırıcı.fit_generator(
        training_set,
        steps_per_epoch=8000//32,
        #bir devrin(epoch) bittiğini ve yeni bir devrin başladığını ilan 
        # etmeden önceki toplam adım sayısı
        # örneklem sayısı ya da örneklem sayısı//batch size
        epochs=3,
        #modeli eğitmek için gerekli devir sayısı
        validation_data=test_set,
        validation_steps=2000//32)


""" 
50 epoch ile benim elde ettigim sonuclar soyle:

Epoch 50/50
250/250 [===- 48s - loss: 0.2532 - acc: 0.8939 - val_loss: 0.3549 - val_acc: 0.8542

"""

""" Yeni bir resim ile tahmin etme. 
Ağımızı eğittik. Şimdi yeni bir resim gösterip bunu tahmin etmesini isteyeceğiz.
"""

import numpy as np
from keras.preprocessing import image
"""resmin yerini ve beklenen boyutu belirtme.
"tahmin" isimli klasorü ve resimleri siz oluşturacaksınız. 
Yapmanız gereken "dataset" klasorunun icine "tahmin" isminde
 bir klasör oluşturmak ve 
 bir kedi ve kopek resmini bu klasore eklemek.
"""
test_image = image.load_img('dataset/tahmin/kedi4.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

#egittigmiz sınıflandırıcıya tahmin yaptırma
sonuc = sınıflandırıcı.predict(test_image)

training_set.class_indices
#sonucu yazdırma 0:kedi 1:kopek
if sonuc[0][0] == 1:
    tahmin = 'kopek'
else:
    tahmin = 'kedi'

print(score,tahmin)
