Bu proje, kullanıcı-film değerlendirme verisini kullanarak derin öğrenme tabanlı Neural Collaborative Filtering (NCF) modeli ile film önerisi yapmayı amaçlamaktadır.
Model, kullanıcının önceki puanlamalarına dayanarak bir film için tahmini puan verir.

ratings.dat: Kullanıcıların filmlere verdikleri puanları içerir.
Format: UserID::MovieID::Rating::Timestamp

movies.dat: Film bilgilerini içerir.
Format: MovieID::Title::Genres
