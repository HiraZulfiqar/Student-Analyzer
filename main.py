import pandas as pd

# Load dataset
df = pd.read_csv("datasets/students.csv")

print("===== Student Dataset =====")
print(df)

#shape
print("\nShape of Dataset:")
print(df.shape)

#no of rows and colum
print("\nColumn Names:")
print(df.columns)

#for information
print("\nInformation:")
df.info()

#for missing values
print("\nMissing Values:")
print(df.isnull().sum())

#________________step 2 _______________(EDA)
#for average marks
print("\nAverage Marks:")
print(df["Marks"].mean())

#for highest score
print("\nhighest Marks:")
print(df["Marks"].max())

#for lowest score
print("\nlowest Marks:")
print(df["Marks"].min())

#for avg study hours 
print("\nAverage Study Hours:")
print(df["StudyHours"].mean())

#for complete statistics (magical command)
print("\nStatistics")
print(df.describe())

#______________step 3_________________(VISUAL EDA)

#count plot (categorical daata)(Count Plot to visualize the frequency (count) of each category in a categorical column.)
import seaborn as sns 
import matplotlib.pyplot as plt 

sns.countplot(data=df, x="Gender")
plt.title("Number of male and female studetns:")
plt.show()

#histogram(Histogram to visualize the distribution of a numerical feature by grouping values into ranges (bins).)

sns.histplot(data=df, x="Marks", bins=5)
plt.title("Distribution of student Marks")
plt.xlabel("Marks")
plt.ylabel("no of studetns")
plt.show()

#BoxPlot(Box Plot to understand the spread of numerical data and detect outliers)
sns.boxplot(data=df, y="Marks")
plt.title("box plot of studetnr Marks")
plt.show()

#ScatterPlot(Scatter Plot to visualize the relationship between two numerical variables.)
sns.scatterplot(data=df, x="StudyHours", y="Marks")
plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()

#heatmap(heatmap is used to visualize the correlation between numerical variables using colors.)
numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()


#__________________step5____________________
#feature engineering(Prepare the data so the machine can understand it and learn from it correctly.)
X = df.drop(columns=["Name", "Marks"])
y = df["Marks"]

print("Features (X):") 
print (X.head())

print("\nTarget (y):")
print (y.head())

#train test split(To divide the dataset into training and testing sets so we can evaluate the model on unseen data.)
  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state= 42
)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#Encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded_data = encoder.fit_transform(X[["Gender", "City"]])


#coloums transformation(Tell me which columns need which transformation, and I'll do everything for you.")
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), ["Gender", "City"]),
        ("num", StandardScaler(), ["Age", "StudyHours"])
    ],
    remainder="passthrough"
)
X_transformed = preprocessor.fit_transform(X)

print(X_transformed)

#pipeline
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LinearRegression())
])
pipe.fit(X_train, y_train)

predictions = pipe.predict(X_test)

print("\nPredicted Marks:")
print(predictions)


print("hello from master")