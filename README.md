# British Airways Predictive Modeling for Proactive Customer Bookings


# Predictive Modeling of Customer Bookings
![Home Page Image](Images/home_page_img.png)

# Introduction
In today's era, customer empowerment, fueled by easy access to information, has transformed the traditional buying cycle. Waiting until customers arrive at the airport to secure bookings is no longer a viable strategy; proactivity is the new imperative. This shift is achievable through strategic data utilization and predictive modeling. The core of success lies in the quality of data powering machine learning algorithms, enabling airlines to stay ahead in a landscape where proactive engagement is paramount.

# Objective
* Explore and prepare dataset for predictive modeling.
* Train a machine learning model for customer booking predictions using a suitable algorithm.
* Evaluate model performance through cross-validation and generate key metrics.
* Create visualizations to interpret variable contributions.

# Tools Used
* Jupyter Notebook (Python)
* Matplotlib
* Seaborn
* Pandas
* Numpy
* pycountry_convert
* scipy.stats
* sklearn.model_selection
* sklearn.ensemble
* sklearn.metrics


# Data Description
- num_passengers = number of passengers travelling
- sales_channel = sales channel booking was made on
- trip_type = trip Type (Round Trip, One Way, Circle Trip)
- purchase_lead = number of days between travel date and booking date
- length_of_stay = number of days spent at destination
- flight_hour = hour of flight departure
- flight_day = day of week of flight departure
-route = origin -> destination flight route
- booking_origin = country from where booking was made
- wants_extra_baggage = if the customer wanted extra baggage in the booking
- wants_preferred_seat = if the customer wanted a preferred seat in the booking
- wants_in_flight_meals = if the customer wanted in-flight meals in the booking
- flight_duration = total duration of flight (in hours)
- booking_complete = flag indicating if the customer completed the booking

## Snapshot of the Data-frame
![Home Page Image](Images/bar_plot.png)

The DataFrame comprises 50,000 entries, with no missing values. It encompasses 14 features or columns, primarily composed of integer and float data types. Specifically, five out of the 14 features are of string data type.

# Data Conversion, Continent Grouping, and Outlier Detection
The categorical columns 'flight_day,' 'sales_channel,' and 'trip_type' underwent encoding, resulting in the creation of new columns with numerical values. Simultaneously, the 'booking_origin' column, with a unique count of 104, was transformed by grouping countries into continents using py_country_converter. Six new columns, one for each continent, were generated and converted into numerical data for analytical and machine learning optimization. Additionally, outlier detection via boxplot analysis identified 'num_of_passengers,' 'purchase_lead,' and 'length_of_stay' as columns with noticeable outliers, necessitating further investigation.

# Exploratory Data Analysis(EDA)

## Outlier Detection with Scatterplots and Z-Score
In the pursuit of identifying potential outliers, a dual approach employing scatterplots and z-score analysis was employed. The scatterplots provided a visual representation of data points, aiding in the identification of patterns and outliers, while the z-score calculation quantified the degree of deviation from the mean, offering a statistical measure to flag potential outliers. Together, these techniques enhance the robustness of outlier detection within the dataset.
![Positive Reviews](Images/positive_sentiment.png)

Upon examining the plot, it is evident that the purchase_lead, representing the duration between booking and travel dates, challenges the straightforward interpretation of outliers. The apparent outliers may not be anomalous, as the nature of this variable involves the temporal span leading to potential variations.

From the plot, a notable concentration of data points falls within a z-score range of 0 to 3, indicating a relatively normal distribution. However, the outliers, characterized by exceptionally high z-scores, signify deviations from this prevalent pattern.

**Extreme Early Bookings:**
- The majority of data points clustering between z-scores of 0 and 3 likely signify bookings made significantly earlier than the conventional booking window.

**Last-Minute Bookings:**
- Outliers with very high z-scores may indicate bookings made very close to the travel date, suggesting last-minute arrangements.

## Days Between Traveling Date and Booking Date Across Continents
The analysis aimed to investigate booking behavior within different continents, specifically focusing on understanding how the distribution of days between booking and traveling varies across these geographical regions.
![Positive Reviews](Images/positive_sentiment.png)


Upon examination of the bar plot, it is evident that the continent with the highest average number of days between traveling date and booking date is **Asia** closely followed by **Oceania** and **Europe**. **South-America**, and **North_America** also exhibits a notable duration, while **Africa** records the least duration on the chart.
*This analysis provides insights into the distribution of days between traveling date and booking date across various continents.*


## Identifying Countries with Extended Booking Windows: Top 20 Rankings
This analysis involves determining the top 20 countries based on their average purchase lead time. This analysis is conducted to understand and highlight countries where travelers tend to plan and book their trips well in advance. The resulting information can provide insights into regional travel habits, economic factors, and cultural influences that impact the duration between booking and travel.

![Positive Reviews](Images/positive_sentiment.png)
In line with the exploration into countries where travelers tend to plan and book well in advance, Malta emerges as the front-runner among the top 20. Bhutan and Afghanistan closely follow, reflecting a trend of extended planning. Interestingly, Turkey stands out with the shortest average purchase lead time in this group, underscoring diverse booking behaviors across these countries. This variation offers valuable insights into the range of regional travel habits and planning preferences.



## Exploration of Data Distribution: Entries Across Continents
This exploration is conducted to gain insights into the distribution of data entries across continents. The resulting information can offer valuable perspectives on regional representation within our dataset, potentially revealing patterns or concentrations that reflect distinct characteristics of each continent.
![Positive Reviews](Images/positive_sentiment.png)


**Noteworthy Business Insights:**

**Regional Dominance:**
- Asia takes the lead as the primary market for flight bookings, with an impressive count surpassing 3000 entries. Oceania closely follows, boasting approximately 2000 entries, suggesting substantial demand in these regions.

**Limited Engagement in Established Markets:**
- Europe and North America, despite being established markets, exhibit relatively lower participation, as reflected in the minimal number of entries.

**Untapped Potential in Emerging Markets:**
- Africa and South America currently show underrepresentation, unveiling untapped potential and providing opportunities for market growth and expansion in these emerging regions.

## Exploring Variable Relationships via Correlation Matrix
![Positive Reviews](Images/positive_sentiment.png)

**Correlation Analysis**
**Positive Correlations:**
- num_passengers has a positive correlation with purchase_lead (0.21) and wants_extra_baggage (0.12).
- length_of_stay is positively correlated with wants_extra_baggage (0.18) and wants_in_flight_meals (0.10).
- flight_day and flight_hour have a positive correlation of 0.02.

**Negative Correlations:**
- length_of_stay has a negative correlation with num_passengers (-0.12).
- purchase_lead is negatively correlated with length_of_stay (-0.08).
- flight_duration is negatively correlated with num_passengers (-0.06) and booking_complete (-0.11).

**Weak Correlations:**
- Many variables show weak correlations (close to zero) with each other.

# Data Preparation
**Train-Test Split of the Dataset:** 80% Training and 20% Testing

# Modeling
Random Forest was selected because it not only effectively predicts customer bookings but also provides invaluable insights into variable contributions, offering a comprehensive and interpretable solution that aligns seamlessly with the dataset exploration, model training, evaluation, and visualization needs.

# Model Prediction and Evaluation
## Classification Report
![Positive Reviews](Images/positive_sentiment.png)

## Classification Metrics by Class
![Positive Reviews](Images/positive_sentiment.png)

## Confusion Matrix
![Positive Reviews](Images/positive_sentiment.png)

Consufion Matrix

True Positive (TP):

99 instances were correctly predicted as class 1.
True Negative (TN):

8394 instances were correctly predicted as class 0.
False Positive (FP):

126 instances were incorrectly predicted as class 1.
False Negative (FN):

1381 instances were incorrectly predicted as class 0.

# Cross Validation 
![Positive Reviews](Images/positive_sentiment.png)

The cross-validation scores [0.8498, 0.8376, 0.7813, 0.3966, 0.6248] exhibit variability, with a mean accuracy of 0.70, providing insights into the model's performance across different subsets of the data.
The accuracy of the model was approximately 0.7 
(Precision) and 0.003 (Recall), showing that this model 
requires more improvement. I suggest adding more 
customer-centric features into the model

# Feature Importances from RandomForest Model
![Positive Reviews](Images/positive_sentiment.png)


- The most important variable in the model was purchase_lead, that is the time between purchase and departure.
- The information regarding the flight, such as flight time and duration, proved to be significant. In contrast, most of the extracted continent information from the booking origin, fully represented by 'Continent_Europe,' 'Continent_South America,' 'Continent_Africa,' and 'Continent_None,' plays a minor role, exerting limited influence on the model's outcome.




The following insights can be gleaned from the areas of importance indicated by the size and frequency of the words:

- **Service:** Its prominence in the word cloud underscores the crucial role that quality service plays in creating positive customer experiences. The emphasis on service points to its perceived value among passengers.
- **Seat:** The visibility of this word suggests that seating both its comfort and the options available is a significant contributor to passenger satisfaction.
- **Food and Drink:** These terms are noticeable and indicate that the food and beverage offerings on board are generally appreciated by customers, contributing positively to their overall experience. 
- **Flight:** The word ‘flight’ appears large and central, indicating that it is a common theme in positive reviews. This likely pertains to the general flight experience, encompassing various elements such as punctuality, smoothness of the journey, and overall comfort.
- **Crew:** The word ‘crew’ is indicative of favorable interactions or experiences with the airline’s staff, highlighting the impact of personal service on the flight experience.
- **Business Class:** The term ‘business class’ is distinct within the cloud, which points to a higher level of satisfaction among passengers traveling in this premium category.
*These insights reflect the airline’s successful areas from the perspective of its customers and can be used to maintain and enhance service quality where it is most appreciated.*

**Visualizing Top 20 Negative Reviews**
![Negative Reviews](Images/negative_sentiment.png)


*The word cloud for Negative Sentiment Reviews displays the words most commonly found in unfavorable British Airways reviews, giving us clues about what might be upsetting customers. The size and prominence of certain words highlight the primary concerns.*

The negative sentiment word cloud sheds light on common concerns expressed by customers in unfavorable reviews:
- **Time and Delayed:** These big words in the cloud tell us that customers often mention being unhappy about flights not being on time. This includes flights leaving late or taking too long.
- **Seat:** Since ‘seat’ pops up a lot in negative reviews, it seems that people are not always happy with their seating. It could be because the seats aren’t comfortable, there’s not enough room, or they had trouble getting the seat they wanted.
- **Food:** ‘Food’ shows up in both good and bad reviews, which means that sometimes the meals on the plane are a hit, and other times people really don’t like them.
- **Staff:** The word ‘staff’ here suggests that sometimes the people working on the plane or at the airport might not be making customers happy, maybe because they aren’t as helpful or friendly as expected.
- **London:** Seeing ‘london’ in the cloud hints that some problems might be linked to flights going to or coming from London, or maybe things happening at London’s airports.

*These insights draw attention to the operational and service areas where the airline could focus its improvements to enhance customer satisfaction and reduce negative feedback.*

# Topic Modeling

**Using CountVectorizer and LatentDirichletAllocation for Topic Modeling**

**CountVectorizer** is a feature extraction technique used in natural language processing (NLP) to convert text data into numerical feature vectors. It is a part of the scikit-learn library in Python. CountVectorizer operates by tokenizing text documents, converting them into a matrix of token counts.

**Latent Dirichlet Allocation (LDA)** is often categorized as a topic modeling technique, it can also be considered a form of feature engineering. It processes the features created by **CountVectorizer** (or another vectorizer) to discover the underlying topics in a text corpus. In doing so, it generates a new set of features related to the topics within the documents. Each document is then described by its distribution of topics, and each topic is characterized by its distribution of words. These topic distributions can be used as features in downstream tasks, such as document classification, and clustering, or as part of a recommendation system.

## Topic Modeling Results

**Latent Dirichlet Allocation (LDA)**
I applied Latent Dirichlet Allocation (LDA) to identify underlying topics within the customer reviews.
**Identified Topics:**
- **Topic #1:** class, flight, food, good, seat, ba, business, service, crew, seats
- **Topic #2:** flight, ba, tokyo, british, airways, tour, service, crew, heathrow, lhr
- **Topic #3:** flight, ba, service, london, time, british, airways, staff, hours, heathrow
- **Topic #4:** flight, ba, airline, hours, time, luggage, help, airport, nov, airways
- **Topic #5:** flight, heading, seats, ba, airways, british, flights, london, glory, airport

**Topic 1:**
![Topic 1](Images/topic_1.png)
- Dominant words: class, flight, food, good, seat, ba, business, service, crew, seats
- Insight: This topic highlights experiences related to flying in business class. Customers appreciate the quality of service, good food, and comfortable seats provided by British Airways. The use of “business” suggests a positive sentiment associated with premium class travel.

**Topic 2:**
![Topic 2](Images/topic_2.png)
- Dominant words: flight, ba, tokyo, british, airways, tour, service, crew, heathrow, lhr.
- Insight: This topic centers around flights to Tokyo with British Airways. Customers discuss the service quality, crew interactions, and possibly experiences at Heathrow Airport (“heathrow” and “lhr”). The term “tour” indicates a potential discussion about travel tours or sightseeing activities associated with these flights.

**Topic 3:**
![Topic 3](Images/topic_3.png)
- Dominant words: flight, ba, service, london, time, british, airways, staff, hours, heathrow.
- Insight: This topic focuses on flights to and from London, particularly Heathrow. Customers discuss the service quality, staff interactions, and possibly the duration of flights (“hours”). The mention of “heathrow” suggests a connection to the London airport.

**Topic 4:**
![Topic 4](Images/topic_4.png)
- Dominant words: flight, ba, airline, hours, time, luggage, help, airport, nov, airways.
- Insight: This topic covers various aspects of flying with British Airways, including discussions about the airline itself, help with luggage, and experiences at the airport. The mention of “nov” might be related to specific events or experiences in November.

**Topic 5:**
![Topic 5](Images/topic_5.png)
- Dominant words: flight, heading, seats, ba, airways, british, flights, london, glory, airport.
- Insight: This topic encompasses discussions about flights with British Airways, focusing on aspects such as seat comfort, heading to different destinations, and experiences at airports. The term “glory” suggests positive or memorable experiences associated with flying.

*Each topic represents a different facet of the flying experience, with Topics 1 and 2 focusing on premium class and flights to Tokyo, Topic 3 on flights to and from London, Topic 4 on general airline experiences, and Topic 5 on overall flight experiences and destinations.*

*The accompanying Jupyter notebook in the 'Notebook' folder provides a detailed narrative of how all analyses were carried out.*


# Conclusion
- In conclusion, this comprehensive analysis of British Airways customer reviews offers a detailed perspective on passenger sentiments and key areas of concern. This wealth of insights equips British Airways with the tools needed for strategic decision-making aimed at boosting customer satisfaction. By proactively addressing identified pain points, capitalizing on positive aspects, and instituting continuous monitoring practices, British Airways can foster a customer-centric approach. This, in turn, has the potential to solidify the airline's standing in the competitive aviation industry. The findings from this analysis serve not only as a roadmap for immediate enhancements but also as a foundational resource for ongoing improvements. It underscores the impactful role of data-driven insights in steering operational excellence and shaping a positive customer experience.


# Recommendation
**Operational Challenges Require Attention:**
- Negative sentiments often revolve around operational issues, especially flight delays. Addressing these challenges can significantly enhance overall service reliability.

**Seating Comfort is Paramount:**
- Both positive and negative sentiments consistently highlight the importance of seating comfort. Emphasizing this aspect can contribute to overall customer satisfaction.

**Premium Class Experiences Stand Out:**
- Positive sentiments associated with business class indicate a strong positive perception. Investing in and promoting premium services can further elevate customer satisfaction.

**Customer Feedback Loop:**
- Establish a robust customer feedback loop for continuous monitoring. Regularly collect and analyze customer feedback to identify evolving trends, address emerging concerns, and maintain a proactive approach to customer satisfaction.

**London-Specific Improvements:**
- Address concerns related to flights to and from London, especially Heathrow Airport. Implement improvements or adjustments that enhance the overall experience for customers traveling through London.

**Continuous Improvement Culture is Vital:**
- The importance of ongoing improvements is emphasized. Establishing a culture of continuous enhancement based on customer feedback ensures sustained positive experiences.
