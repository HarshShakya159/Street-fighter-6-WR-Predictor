# Harsh Shakya
# Street Fighter 6 Win Rate Predictor
# Using the 6 main mechanics to win
# Drive Impacts, Perfect Parries, OD Dps, Level 1-3 Supers
# Not the most complex but I'm learning
# harshshakya765@gmail.com

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


np.random.seed(50)
n_samples = 1000

num_of_drive_impacts = np.random.randint(1, 5, n_samples)
num_of_perfect_parries = np.random.randint(3, 10, n_samples)
num_of_od_dps = np.random.randint(5, 10, n_samples)
num_of_l1 = np.random.randint(1, 5, n_samples)
num_of_l2 = np.random.randint(1, 3, n_samples)
num_of_l3 = np.random.randint(0, 2, n_samples)


dataframe = pd.DataFrame({
    'Drive Impacts': num_of_drive_impacts,
    'Perfect Parries': num_of_perfect_parries,
    'OD Dps': num_of_od_dps,
    'Level 1': num_of_l1,
    'Level 2': num_of_l2,
    'Level 3': num_of_l3
})


base_skill = 15
match_win_rate = (
    base_skill +
    dataframe['Perfect Parries'] * 2 + # 6-20 points
    dataframe['Drive Impacts'] * 3 + # 3-15 points
    dataframe['OD Dps'] * 1 + # 5-10 points
    dataframe['Level 1'] * 2 + # 2-10 points
    dataframe['Level 2'] * 4 + # 4-12 points
    dataframe['Level 3'] * 6 + # 0-12 points
    # adding player error
    np.random.normal(0, 8, n_samples)
)
dataframe['Win Rate'] = np.clip(match_win_rate, 0, 100)

print(f'{len(dataframe)} and Columsn: {dataframe.columns}')
print(dataframe.head())
print(dataframe.info())
print(dataframe.describe())
print(dataframe.isnull().sum())


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
correlation_matrix = dataframe.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Which Stats Help You Win?')


plt.subplot(1, 2, 2)
plt.hist(dataframe['Win Rate'], bins=25, alpha=0.7, color='red')
plt.title('Player Win Rate Distribution')
plt.xlabel('Win Rate (%)')
plt.ylabel('Number of Players')
plt.tight_layout()
plt.show()


features = ['Drive Impacts', 'Perfect Parries', 'OD Dps', 'Level 1', 'Level 2', 'Level 3']
X = dataframe[features]
y = dataframe['Win Rate']
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")


print("X (features) looks like:")
print(X.head())
print(f"X dimensions: {X.ndim}D")

print("\ny (target) looks like:")
print(y.head())
print(f"y dimensions: {y.ndim}D")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=50)
}


trained_models = {}
for name, model in models.items():
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        trained_models[name] = (model, True)
    else:
        model.fit(X_train, y_train)
        trained_models[name] = (model, False)

    print(f'{name} trained')


results = {}

for name, (model, use_scaled) in trained_models.items():
    print(f"\nEvaluating {name}:")

    # Make predictions
    if use_scaled:
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {
        'predictions': y_pred,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

plt.figure(figsize=(15, 6))

for i, (name, result) in enumerate(results.items(), 1):
    plt.subplot(1, 2, i)

    plt.scatter(y_test, result['predictions'], alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Win Rate (%)')
    plt.ylabel('Predicted Win Rate (%)')
    plt.title(f'{name}\nR² = {result["r2"]:.4f}')

    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

rf_model = trained_models['Random Forest'][0]
feature_importance = rf_model.feature_importances_

plt.figure(figsize=(10, 6))
importance_df = pd.DataFrame({
    'feature': features,
    'importance': feature_importance
}).sort_values('importance', ascending=True)

plt.barh(importance_df['feature'], importance_df['importance'])
plt.title('Most Important Combat Stats for Winning')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

print("Feature Importance Ranking:")
for feature, importance in zip(importance_df['feature'][::-1], importance_df['importance'][::-1]):
    print(f"  {feature}: {importance:.4f}")

num_players = 3
np.random.seed(None) # new seed for new players to keep it random
new_player = pd.DataFrame({
    'Drive Impacts': np.random.randint(1, 5, num_players),
    'Perfect Parries': np.random.randint(3, 10, num_players),
    'OD Dps': np.random.randint(5, 10, num_players),
    'Level 1': np.random.randint(1, 5, num_players),
    'Level 2': np.random.randint(1, 3, num_players),
    'Level 3': np.random.randint(0, 2, num_players)
})
print("New player combat stats:")
print(new_player)

best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
best_model, use_scaled = trained_models[best_model_name]

print(f"\nUsing best model: {best_model_name}")

for i in range(len(new_player)):
    player_data = new_player.iloc[[i]]

    if use_scaled:
        player_scaled = scaler.transform(player_data)
        predicted_win_rate = best_model.predict(player_scaled)[0]
    else:
        predicted_win_rate = best_model.predict(player_data)[0]

    print(f"Player {i + 1} stats: {player_data.iloc[0].to_dict()}")
    print(f"Predicted win rate: {predicted_win_rate:.1f}%")

if use_scaled:
    new_player_scaled = scaler.transform(new_player)
    predicted_win_rate = best_model.predict(new_player_scaled)[0]
else:
    predicted_win_rate = best_model.predict(new_player)[0]

print(f"Predicted win rate: {predicted_win_rate:.1f}%")

if predicted_win_rate >= 80:
   skill_level = "Punk Level"
elif predicted_win_rate >= 65:
   skill_level = "Sweaty Player"
elif predicted_win_rate >= 50:
   skill_level = "Wannabe Pro"
elif predicted_win_rate >= 35:
   skill_level = "Jus Mid"
elif predicted_win_rate >= 20:
   skill_level = "Trash Player"
else:
   skill_level = "Uninstall"

print(f"Skill Assessment: {skill_level}")
