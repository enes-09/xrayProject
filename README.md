X-RAY GÖRÜNTÜ ANALİZ PROJESİ - GEREKLİ YAZILIMLAR VE ARAÇLAR

KATKI VERENLER
- [Enes Kulpu](https://github.com/enes-09)
- [Berkay Özer](https://github.com/theberkayozer)

Bu projenin kaynak kodlarını görüntülemek, düzenlemek ve çalıştırmak için aşağıdaki yazılımlara ihtiyaç duyulmaktadır:

1. Visual Studio Code (VS Code)
   - Açıklama: Frontend (React) kodlarını düzenlemek ve geliştirmek için önerilen kod editörüdür.

2. Node.js ve npm
   - Açıklama: Frontend (React) projesinin paket bağımlılıklarını yönetmek ve uygulamayı yerel sunucuda çalıştırmak için gereklidir.

3. IntelliJ IDEA (veya Eclipse)
   - Açıklama: Backend (Spring Boot) projesini açmak, derlemek ve yönetmek için kullanılan Java IDE'sidir.

4. Java Development Kit (JDK 17 veya üzeri)
   - Açıklama: Backend tarafındaki Java Spring Boot uygulamasının çalışabilmesi için gereken temel Java geliştirme kitidir.

5. Google Colab
   - Açıklama: Derin öğrenme modellerinin (PyTorch) yüksek performanslı GPU üzerinde çalıştırılması için kullanılan bulut tabanlı Python geliştirme ortamıdır. (Tarayıcı üzerinden erişilir, kurulum gerektirmez).

6. Ngrok
   - Açıklama: Google Colab üzerinde çalışan AI servisini (localhost), yerel bilgisayarda çalışan Backend uygulamasına bağlamak (public URL oluşturmak) için kullanılır.

7. Web Tarayıcısı (Google Chrome, Microsoft Edge vb.)
   - Açıklama: Web uygulamasının arayüzünü görüntülemek ve Google Colab ortamına erişmek için kullanılır.
   

X-RAY GÖRÜNTÜ ANALİZ PROJESİ - ÇALIŞTIRMA VE DERLEME ADIMLARI

Projenin sorunsuz çalışması için aşağıdaki adımları sırasıyla takip ediniz. Sistem 3 ana parçadan oluşmaktadır: AI Servisi (Colab), Backend (Spring Boot) ve Frontend (React).

ADIM 1: AI SERVİSİNİN (GOOGLE COLAB) BAŞLATILMASI

1. Google Colab sayfasını açın (https://colab.research.google.com/).
2. Proje klasöründeki "backend_script_colab.py" dosyasının içeriğini kopyalayıp yeni bir not defterine yapıştırın. (Sadece bu kod yeterlidir).
3. Google Drive Bağlantısı:
   - Eğer model dosyalarınız (.pth) Google Drive'da ise, kodun başına şu satırları ekleyerek Drive'ı bağlayın:
     
     from google.colab import drive
     drive.mount('/content/drive')

   - Kod içerisindeki 'MODEL_CONFIGS' bölümünde yer alan 'file_path' kısımlarını, Drive'ınızdaki dosya yollarına göre güncelleyin.
     Örn: 'file_path': '/content/drive/MyDrive/XRay_Models/best_swin_t_fold_2.pth'

4. Menüden "Çalışma Zamanı" > "Çalışma Zamanı Türünü Değiştir" diyerek donanım hızlandırıcıyı "T4 GPU" olarak seçin.
5. Kodu çalıştırın. Drive onayı isteyecektir, onay verin.
6. Kodun çıktısında "LİNKİNİZ: https://xxxx.ngrok-free.app/predict" şeklinde bir URL göreceksiniz.
   -> BU URL'İ KOPYALAYIN.

ADIM 2: BACKEND (SPRING BOOT) PROJESİNİN DERLENMESİ VE ÇALIŞTIRILMASI

1. Proje klasörü altındaki "Backend" klasörüne gidin.
2. `src/main/resources/application.properties` dosyasını açın.
3. `ai.service.url` değerini, ADIM 1'de kopyaladığınız ngrok linki ile güncelleyin:
   Örn: ai.service.url=https://xxxx-xx-xx-xx.ngrok-free.app/predict
   (Kaydetmeyi unutmayın).

4. Terminal veya Komut İstemi'ni (CMD) "Backend" klasöründe açın.
5. Aşağıdaki komutu çalıştırarak projeyi derleyin ve başlatın:
   
   mvn spring-boot:run
   (Eğer Maven yüklü değilse veya IDE kullanıyorsanız, IntelliJ/Eclipse içinde "Run" butonuna basarak Main class'ı çalıştırabilirsiniz).
   
7. Konsolda "Started Application in ... seconds" yazısını gördüğünüzde backend hazırdır (Varsayılan Port: 8080).

ADIM 3: FRONTEND (REACT) PROJESİNİN DERLENMESİ VE ÇALIŞTIRILMASI

1. Yeni bir Terminal penceresi açın ve proje klasörü altındaki "Frontend" klasörüne gidin.
2. Gerekli paketleri yüklemek için şu komutu çalıştırın (Sadece ilk kurulumda gereklidir):
   
   npm install

3. Uygulamayı başlatmak için şu komutu çalıştırın:
   
   npm run dev

4. Terminalde "Local: http://localhost:5173" gibi bir adres göreceksiniz.
5. Tarayıcınızda bu adresi açarak uygulamayı kullanmaya başlayabilirsiniz.
=================================================================
- Kod çalıştırıldığında .pth modelleri sizin belirttiğiniz Google Drive yolundan çekilecektir.
- Tarayıcıda http://localhost:5173 adresine gidildiğinde arayüz gelmelidir.
- Model listesi ve analiz sonuçları için Backend ve Colab servisinin açık olması şarttır.
- İşiniz bittiğinde terminalleri kapatarak servisleri durdurabilirsiniz.
   
<img width="1911" height="853" alt="image" src="https://github.com/user-attachments/assets/620925ad-67bc-408c-a39a-84ba0b874010" />
<img width="1915" height="858" alt="image" src="https://github.com/user-attachments/assets/3f4a854e-1ac2-43a0-9c8d-bdf29497870a" />

