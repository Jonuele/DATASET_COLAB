import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
import missingno as msno
import contextily as ctx
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge


listings=pd.read_csv(r"C:\Users\Usuario\OneDrive\Coder\Data Science\Prediccion de Precios alquiler temporario\Get data\listings.csv",header=0)
tabla_dolar=pd.read_excel(r"C:\Users\Usuario\OneDrive\Coder\Data Science\Prediccion de Precios alquiler temporario\Get data\Tabla Dolar BNA.xlsx",header=0)
#print(tabla_dolar.dtypes)
print(tabla_dolar.to_csv('Tabla_Dolar_BNA.csv', index=False))
##Acomodamos el Archivo TABLA DOLAR, ya que vamos a utilizarlo para expresar los precios por noche de cada propiedad##

tabla_dolar['Fecha']= pd.to_datetime(tabla_dolar['Fecha'],format='%Y-%m-%d', errors='coerce')
tabla_dolar=tabla_dolar.rename(columns={'Tipo de Cambio de Referencia - en Pesos - por Dólar':'Tipo de Cambio'})

#print(tabla_dolar.info())
# print(listings.columns)
# print(listings.info())
# print(listings.shape)

##Iniciamos el proceso de analisis y limpieza de datos del archivo Listings##
#print(listings.columns)
listings_1=listings[['id','last_scraped','host_id','host_name', 'host_since','host_response_time','host_is_superhost', 'host_listings_count',
       'neighbourhood_cleansed', 'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms',
       'bedrooms', 'beds', 'amenities', 'price',
       'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value']]

#print(listings_1.to_csv('Listings.csv', index=False))
# print(listings_1.info())
# print(listings_1.shape)
                            # Data columns (total 27 columns):
                            #  #   Column                       Non-Null Count  Dtype
                            # ---  ------                       --------------  -----
                            #  0   id                           37035 non-null  int64
                            #  1   scrape_id                    37035 non-null  int64
                            #  2   host_id                      37035 non-null  int64
                            #  3   host_name                    37033 non-null  object
                            #  4   host_since                   37033 non-null  object
                            #  5   host_response_time           29654 non-null  object
                            #  6   host_is_superhost            34839 non-null  object
                            #  7   host_listings_count          37033 non-null  float64
                            #  8   neighbourhood_cleansed       37035 non-null  object
                            #  9   latitude                     37035 non-null  float64
                            #  10  longitude                    37035 non-null  float64
                            #  11  property_type                37035 non-null  object
                            #  12  room_type                    37035 non-null  object
                            #  13  accommodates                 37035 non-null  int64
                            #  14  bathrooms                    34168 non-null  float64
                            #  15  bedrooms                     36422 non-null  float64
                            #  16  beds                         34164 non-null  float64
                            #  17  amenities                    37035 non-null  object
                            #  18  price                        34005 non-null  object
                            #  19  number_of_reviews            37035 non-null  int64
                            #  20  review_scores_rating         29760 non-null  float64
                            #  21  review_scores_accuracy       29736 non-null  float64
                            #  22  review_scores_cleanliness    29736 non-null  float64
                            #  23  review_scores_checkin        29736 non-null  float64
                            #  24  review_scores_communication  29737 non-null  float64
                            #  25  review_scores_location       29737 non-null  float64
                            #  26  review_scores_value          29736 non-null  float64
#(37035, 27)

##Reordenamos los Dtypes de cada columna
listings_1=listings_1.copy()              
listings_1['last_scraped']=listings_1['last_scraped'].replace(np.nan,"")
listings_1['last_scraped']=pd.to_datetime(listings_1['last_scraped'], format='%Y-%m-%d', errors='coerce')

listings_1['host_since']=listings_1['host_since'].replace(np.nan,"")
listings_1['host_since']=pd.to_datetime(listings_1['host_since'], format='%Y-%m-%d', errors='coerce')

# listings_1.loc[:, 'bathrooms'] = listings_1['bathrooms'].replace(np.nan, 0).replace("", 0)
# listings_1 = listings_1[listings_1['bathrooms'] != ""]
# listings_1.loc[:, 'bathrooms'] = listings_1['bathrooms'].astype(int)

listings_1['bathrooms'] = listings_1['bathrooms'].fillna(0).round().astype('int64')
listings_1 = listings_1.dropna(subset=['bathrooms'])
listings_1['bathrooms'] = listings_1['bathrooms'].round().astype('int64')

listings_1['bedrooms'] = listings_1['bedrooms'].fillna(0).round().astype('int64')
listings_1 = listings_1.dropna(subset=['bedrooms'])
listings_1['bedrooms'] = listings_1['bedrooms'].round().astype('int64')

listings_1['beds'] = listings_1['beds'].fillna(0).round().astype('int64')
listings_1 = listings_1.dropna(subset=['beds'])
listings_1['beds'] = listings_1['beds'].round().astype('int64')


listings_1['price']=listings_1['price'].replace({r'\$':''}, regex=True)
listings_1['price']=listings_1['price'].replace({',':''}, regex=True)
listings_1['price'] = pd.to_numeric(listings_1['price'], errors='coerce')
listings_1['price']=listings_1['price'].fillna(0)
#listings_1['price'] = listings_1['price'].round().astype('int64')
#print(listings_1.info())


listings_1=listings_1[(listings_1['price'] != 0) | ((listings_1['price'] != 0) & (listings_1['host_id'] == 374872974))]
listings_1=listings_1[listings_1['neighbourhood_cleansed'].isin(['Palermo','Belgrano','Recoleta','San Telmo','Areco','Balvanera','Villa Crespo','Saavedra'])]
# print(listings_1.columns)
#print(listings_1.info())
# print(listings_1.shape)
#print(listings_1.to_csv('Listings.csv', index=False))

                                # Data columns (total 27 columns):
                                #  #   Column                       Non-Null Count  Dtype
                                # ---  ------                       --------------  -----
                                #  0   id                           21373 non-null  int64
                                #  1   last_scraped                 21373 non-null  object
                                #  2   host_id                      21373 non-null  int64
                                #  3   host_name                    21372 non-null  object
                                #  4   host_since                   21372 non-null  object
                                #  5   host_response_time           18147 non-null  object
                                #  6   host_is_superhost            19919 non-null  object
                                #  7   host_listings_count          21372 non-null  float64
                                #  8   neighbourhood_cleansed       21373 non-null  object
                                #  9   latitude                     21373 non-null  float64
                                #  10  longitude                    21373 non-null  float64
                                #  11  property_type                21373 non-null  object
                                #  12  room_type                    21373 non-null  object
                                #  13  accommodates                 21373 non-null  int64
                                #  14  bathrooms                    21373 non-null  float64
                                #  15  bedrooms                     21373 non-null  float64
                                #  16  beds                         21373 non-null  float64
                                #  17  amenities                    21373 non-null  object
                                #  18  price                        21373 non-null  float64
                                #  19  number_of_reviews            21373 non-null  int64
                                #  20  review_scores_rating         17911 non-null  float64
                                #  21  review_scores_accuracy       17896 non-null  float64
                                #  22  review_scores_cleanliness    17897 non-null  float64
                                #  23  review_scores_checkin        17897 non-null  float64
                                #  24  review_scores_communication  17897 non-null  float64
                                #  25  review_scores_location       17897 non-null  float64
                                #  26  review_scores_value          17896 non-null  float64
                                # dtypes: float64(14), int64(4), object(9)
                                # memory usage: 4.6+ MB
                                # None
                                # (21373, 27)
                                
                                


#Despues del analisis conseguimos la columna en pesos y realizamos un merge para colocar el tipo de cambio

merge=pd.merge(listings_1, tabla_dolar, left_on= 'last_scraped', right_on= 'Fecha', how= 'left')
#merge=merge.drop(columns='Fecha')
#print(merge.info())
                            # Data columns (total 28 columns):
                            #  #   Column                                               Non-Null Count  Dtype
                            # ---  ------                                               --------------  -----
                            #  0   id                                                   0 non-null      int64
                            #  1   last_scraped                                         0 non-null      datetime64[ns]
                            #  2   host_id                                              0 non-null      int64
                            #  3   host_name                                            0 non-null      object
                            #  4   host_since                                           0 non-null      datetime64[ns]
                            #  5   host_response_time                                   0 non-null      object
                            #  6   host_is_superhost                                    0 non-null      object
                            #  7   host_listings_count                                  0 non-null      float64
                            #  8   neighbourhood_cleansed                               0 non-null      object
                            #  9   latitude                                             0 non-null      float64
                            #  10  longitude                                            0 non-null      float64
                            #  11  property_type                                        0 non-null      object
                            #  12  room_type                                            0 non-null      object
                            #  13  accommodates                                         0 non-null      int64
                            #  14  bathrooms                                            0 non-null      int64
                            #  15  bedrooms                                             0 non-null      int64
                            #  16  beds                                                 0 non-null      int64
                            #  17  amenities                                            0 non-null      object
                            #  18  price                                                0 non-null      float64
                            #  19  number_of_reviews                                    0 non-null      int64
                            #  20  review_scores_rating                                 0 non-null      float64
                            #  21  review_scores_accuracy                               0 non-null      float64
                            #  22  review_scores_cleanliness                            0 non-null      float64
                            #  23  review_scores_checkin                                0 non-null      float64
                            #  24  review_scores_communication                          0 non-null      float64
                            #  25  review_scores_location                               0 non-null      float64
                            #  26  review_scores_value                                  0 non-null      float64
                            #  27  Tipo de Cambio                                       0 non-null      float64
                            # dtypes: datetime64[ns](2), float64(12), int64(7), object(7)
                            # memory usage: 132.0+ bytes




###Grafico tipo de propiedades###

grafics_1=merge['room_type'].value_counts()
grafics_1=grafics_1.reset_index()
grafics_1.columns=['room_type','cantidad']
#print(grafics_1)

# sns.barplot(data=grafics_1, x='room_type',y='cantidad')
# plt.xlabel('Tipo de propiedad')
# plt.ylabel('Cantidad')
# plt.title('Tipos de propiedades y cantidades')
#plt.show()



###Grafico: Distribucion de propiedades (barrios) ###

grafics_2=merge['neighbourhood_cleansed'].value_counts()
grafics_2=grafics_2.reset_index()
grafics_2.columns=['neighbourhood_cleansed','cantidad']
#print(grafics_2)

# sns.barplot(data=grafics_2, x='neighbourhood_cleansed',y='cantidad')
# plt.xlabel('Ubicacion')
# plt.ylabel('Cantidad')
# plt.title('Distribucion de propiedades por barrio en CABA')
# plt.show()


###Grafico minimo de noches ###

#grafics_3 = listings.groupby('minimum_nights')['host_name'].count()
#grafics_3=grafics_3.reset_index()
#grafics_3.columns=['minimum_nights','host_name']

        #print(grafics_3)
                        #     minimum_nights  host_name
                        # 0                1       5712
                        # 1                2       5718
                        # 2                3       4807
                        # 3                4       1473
                        # 4                5       1079
                        # ..             ...        ...

# sns.scatterplot(data=grafics_3, x='minimum_nights', y='host_name')
# plt.xlabel('Noches minimas ofrecidas')
# plt.ylabel('Cantidad de Oferentes')
# plt.title('Dispercion de noches minimas requeridas y sus oferta')
# plt.show()


###Grafico de precio promedio por propiedad###

###Se puede observar un error en la evalucacion de los decimales en la columna "price",
# por lo tanto dividimos por 1000 para tener mas logica en la muestra, aunque de todas formas es algo a relevar y buscar la info correcta
# y de esta forma obtener correctamente outliers.

#La solucion fue usar el mismo dataset con los precios expresados en pesos, ajustando al tipo de cambio de la fecha para poder visualizarlo en dolares

grafics_4=merge
grafics_4['price_update']=grafics_4['price']/grafics_4['Tipo de Cambio']
grafics_4['price_update']=grafics_4['price_update'].fillna(0)

#print(grafics_4.describe())



###Grafico mapa de densidad por propiedad

#Podemos ajustar algunas propiedades que se escapan de la densidad del mapa, y de la zona de concentracion

grafics_5=listings_1[['latitude','longitude','neighbourhood_cleansed','room_type']]

# ax= sns.scatterplot(data=grafics_5, x='longitude',alpha=0.1, y='latitude')
# plt.xlabel('longitud')
# plt.ylabel('latitud')
# plt.title('Concentracion de departamentos en CABA')

# ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron)
# plt.show()



#Hacemos un analisis para ver cuales son los precios minimos y maximos de la noche segun cada Barrio de CABA


New_grafics_4=grafics_4
unique= New_grafics_4['neighbourhood_cleansed'].unique()
for x in unique:
        Q1 = New_grafics_4[New_grafics_4['neighbourhood_cleansed'] == x]['price_update'].quantile(0.25)
        Q3 = New_grafics_4[New_grafics_4['neighbourhood_cleansed'] == x]['price_update'].quantile(0.75)
        IQR = Q3 - Q1
        min_price= New_grafics_4[New_grafics_4['neighbourhood_cleansed'] == x]['price_update'].quantile(0.01) #evitamos valores negativos   
        max_price= Q3 + (1.5 * IQR)
        #print(x)
        #print("El precio minimo es {}, mientras que el precio maximo es {}". format(min_price,max_price))
        


#Definimos limitar el Dataset en un minimo de USD 15 y maximo USD 200 la noche ya que por fuera de ese rango nuestros clientes no estan dispuestos a abonar.
#porque no cumple los requisitos minimos o porque los supera.

New_grafics_4=New_grafics_4[(New_grafics_4['price_update'] >= 15) & (New_grafics_4['price_update'] <= 200)]

conca_1=New_grafics_4


# # Convertimos aquellas variables categóricas en dummies

data_encoded = pd.get_dummies(conca_1, columns=['neighbourhood_cleansed','room_type'], drop_first=True)

conca_2=data_encoded[['price_update','bathrooms', 'bedrooms', 'beds',
        'neighbourhood_cleansed_Belgrano', 'neighbourhood_cleansed_Palermo',
        'neighbourhood_cleansed_Recoleta', 'neighbourhood_cleansed_Saavedra',
        'neighbourhood_cleansed_San Telmo',
        'neighbourhood_cleansed_Villa Crespo', 'room_type_Hotel room',
        'room_type_Private room', 'room_type_Shared room']]

correlation_matrix = conca_2.corr()


#sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm', linewidths=3, linecolor='black')
plt.figure(figsize=(10, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de Correlación')
plt.show()


#Preparamos el modelo para realizar una Regresion Lineal y evaluarlo

X = conca_2[['bathrooms', 'bedrooms', 'beds',
        'neighbourhood_cleansed_Belgrano', 'neighbourhood_cleansed_Palermo',
        'neighbourhood_cleansed_Recoleta', 'neighbourhood_cleansed_Saavedra',
        'neighbourhood_cleansed_San Telmo',
        'neighbourhood_cleansed_Villa Crespo', 'room_type_Hotel room',
        'room_type_Private room', 'room_type_Shared room']]

y = conca_2['price_update']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()  # Aumentar max_iter si es necesario
model.fit(X_train, y_train)

# # # # # Evaluar el modelo # # # #
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)


# print("Error cuadratico Medio:", mse)
# print("Devio estandar del modelo:", rmse)
# print("Coeficiente:", r2)


# Error cuadratico Medio: 514.2500972689911
# Devio estandar del modelo: 22.677083085551175
# Coeficiente: 0.3207262356787699

#Despues de ver los resultados, llego a la conclusion que aun no es un modelo optimo, ya que tiene un coeficiente de 0.32 y gran desvio. debemos realizar
#nuevos analisis que nos ayuden a mejorar la precision del modelo.







#Datos de ayuda

# ['id', 'last_scraped', 'host_id', 'host_name', 'host_since',
#        'host_response_time', 'host_is_superhost', 'host_listings_count',
#        'latitude', 'longitude', 'property_type', 'accommodates', 'bathrooms',
#        'bedrooms', 'beds', 'amenities', 'price', 'number_of_reviews',
#        'review_scores_rating', 'review_scores_accuracy',
#        'review_scores_cleanliness', 'review_scores_checkin',
#        'review_scores_communication', 'review_scores_location',
#        'review_scores_value', 'Fecha', 'Tipo de Cambio', 'price_update',
#        'neighbourhood_cleansed_Belgrano', 'neighbourhood_cleansed_Palermo',
#        'neighbourhood_cleansed_Recoleta', 'neighbourhood_cleansed_Saavedra',
#        'neighbourhood_cleansed_San Telmo',
#        'neighbourhood_cleansed_Villa Crespo', 'room_type_Hotel room',
#        'room_type_Private room', 'room_type_Shared room'

#Index(['id', 'last_scraped', 'host_id', 'host_name', 'host_since',
#        'host_response_time', 'host_is_superhost', 'host_listings_count',
#        'neighbourhood_cleansed', 'latitude', 'longitude', 'property_type',
#        'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
#        'amenities', 'price', 'number_of_reviews', 'review_scores_rating',
#        'review_scores_accuracy', 'review_scores_cleanliness',
#        'review_scores_checkin', 'review_scores_communication',
#        'review_scores_location', 'review_scores_value', 'Fecha',
#        'Tipo de Cambio', 'price_update'],