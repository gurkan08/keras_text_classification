# keras Turkish text classification service (7 class)

    docker build -t keras_turkish_text_classification .

    docker run -it -p 8000:8000 keras_turkish_text_classification

# example request

    API endpoint (GET, POST): http://localhost:8000/text-classification-service/

    {

        "text": "İnşaatı tamamlandığından beri boş olan bu bina bu kez eski CHP'li Mustafa Sarıgül ile gündemde. Sarıgül'ün yeni bir parti kuracağı ve genel merkez olarak da bu binayı seçtiği ileri sürüldü ama Sarıgül bu iddiaları yalanladı."

    }


    {
        "result": {
            "siyaset": 0.36192184686660767,
            "diger": 0.34304600954055786,
            "magazin": 0.10055827349424362,
            "ekonomi": 0.0953732505440712,
            "spor": 0.045774903148412704,
            "teknoloji": 0.03289984166622162,
            "saglik": 0.020425908267498016
        }
    }

    {
    
        "text": "Survivor 2020 ile adını duyuran Ardahan Uzkanbaş, Bu Tarz Benim yarışmasıyla tanınan Tuğçe Ergişi'yle aşk yaşamaya başladı. Çift, ilk pozlarını da Instagram hesaplarından yayınladı."
        
    }
    
    {
        "result": {
            "magazin": 0.7542529702186584,
            "diger": 0.17707431316375732,
            "spor": 0.02132243476808071,
            "teknoloji": 0.017378076910972595,
            "siyaset": 0.011529525741934776,
            "saglik": 0.011078386567533016,
            "ekonomi": 0.0073642730712890625
        }
    }


    {
        "text": "Fenerbahçe'ye yakınlığıyla bilinen Rıdvan Dilmen, Jailson'un kulüpten ayrılmaya yakın olduğunu açıkladı. Fenerbahçe'nin oyuncudan 4.5-5 milyon euro arası bir bonservis beklediğini söyleyen Dilmen, duyum aldığını belirtti."
    }

    {
        "result": {
            "spor": 0.7072558403015137,
            "diger": 0.1680379956960678,
            "ekonomi": 0.044288270175457,
            "magazin": 0.032457754015922546,
            "siyaset": 0.024948155507445335,
            "teknoloji": 0.017092445865273476,
            "saglik": 0.005919513292610645
        }
    }
    
    {
        "text": "Covid-19 pandemisinden etkilenen iş yerleri için şartları kolaylaştırılan kısa çalışma ödeneği alanların sayısı ağustos ayında da azaldı. Nisan ve mayıs aylarında 3 milyonun üzerinde olan ödenek alanların sayısı hazirandaki normalleşmeden itibaren azalmaya başlamıştı. Ağustos ayında 562 bin 557 kişi daha azalarak 1.2 milyon kişiye geriledi. Habertürk’ten Ahmet Kıvanç’ın haberi"
    }
    
    {
        "result": {
            "ekonomi": 0.4441114366054535,
            "diger": 0.25781768560409546,
            "siyaset": 0.10051736980676651,
            "teknoloji": 0.06665883213281631,
            "magazin": 0.05796588957309723,
            "saglik": 0.03928801044821739,
            "spor": 0.03364083170890808
        }
    }
    
    {
        "text": "Samsung Galaxy Z Fold2 özellikleri Cihazın ekranına bakacak olursak bizi katlanmadan kullanılan 6.2 inç AMOLED bir ekran ekran karşılıyor. Ana ekran ise Infinity-O teknolojilisine sahip ve  7.6 inç büyüklüğünde. AMOLED ekranla gelen bu cihaz 120hz yenileme hızını bünyesinde bulunduruyor. Adaptif olan bu yenileme hızı kullanıcıya güzel bir deneyimi yaşatacaktı"
    }
    
    {
        "result": {
            "teknoloji": 0.32479655742645264,
            "diger": 0.2230416238307953,
            "saglik": 0.18649935722351074,
            "ekonomi": 0.15828953683376312,
            "magazin": 0.05206535756587982,
            "siyaset": 0.03410032391548157,
            "spor": 0.021207226440310478
        }
    }
    
    {
        "text": "Kalsiyum vücudumuzun en çok ihtiyaç duyduğu mineral. Süt ürünleri ve yeşil sebzelerde bol miktarda bulunan kalsiyum, kemikler için olmazsa olmaz bir mineral. Bu nedenle kalsiyum içeren besinler tüketmekte fayda var. Peki hangi besinlerde kalsiyum var? Kalsiyum açısından zengin besinler hangileri? İşte kemikleri güçlendiren kalsiyum zengini 11 mucize besin."
    }
    
    {
        "result": {
            "saglik": 0.8295288681983948,
            "diger": 0.0984903872013092,
            "teknoloji": 0.02778933197259903,
            "ekonomi": 0.016082430258393288,
            "magazin": 0.015515281818807125,
            "siyaset": 0.00970554817467928,
            "spor": 0.0028880941681563854
        }
    }
    
    
    
    
    


