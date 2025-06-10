# src/06_analysis_and_plots.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings

# Suppress warnings for cleaner output, especially from Cartopy/Matplotlib
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def plot_timeseries_for_routes(data, routes_to_plot, output_path):
    """
    Plots passenger volume time-series for a list of specified routes in a 2x3 grid.

    Args:
        data (pd.DataFrame): The full dataset containing predictions.
        routes_to_plot (list of tuples): A list of (origin, destination) tuples.
        output_path (str): Path to save the output plot.
    """
    print("Generating 2x3 time-series grid plot for specified routes...")
    
    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 12))
    # Flatten the 2D array of axes for easy iteration
    axes = axes.flatten()
    
    # Ensure 'Date' is a datetime object for plotting
    data['Date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
    
    for i, (orig, dest) in enumerate(routes_to_plot):
        # If we have more routes than subplots, stop to avoid errors
        if i >= len(axes):
            print(f"Warning: More routes than available subplots ({len(axes)}). "
                  f"Skipping route: {orig} -> {dest}")
            break
            
        ax = axes[i]
        
        # Filter data for the specific route and sum passengers per month
        route_data = data[
            (data['Orig.Country'] == orig) & (data['Dest.Country'] == dest)
        ].groupby('Date')['Reported.Est..Pax'].sum().reset_index()
        
        if route_data.empty:
            print(f"Warning: No data found for route: {orig} -> {dest}")
            ax.set_title(f'Passenger Volume: {orig} to {dest} (No Data)')
            ax.grid(True)
            continue
            
        # Plot the data
        ax.plot(route_data['Date'], route_data['Reported.Est..Pax'], linestyle='-')
        ax.set_title(f'Passenger Volume: {orig} → {dest}')
        ax.set_xlabel('Months and Year')
        ax.set_ylabel('Estimated Passengers')
        ax.grid(True)
        ax.legend([f'{orig} → {dest}'], loc='upper right') # Add a legend
        
    # If there are any unused subplots, hide them
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig) # Close the figure to free up memory
    print(f"Time-series grid plot saved to {output_path}")

def plot_connection_map(ax, connections, value_col, cmap_name, title):
    """
    Plots flight connections on a given Cartopy axes.
    """
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='#d3d3d3')
    ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='#add8e6')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Filter out rows where coordinates are missing
    connections = connections.dropna(subset=['long_orig', 'lat_orig', 'long_dest', 'lat_dest'])
    if connections.empty:
        print(f"Warning: No valid connection data to plot for '{title}'.")
        ax.set_title(title, fontsize=10)
        return

    # Normalize passenger counts for color mapping
    norm = Normalize(vmin=connections[value_col].min(), vmax=connections[value_col].max())
    cmap = plt.get_cmap(cmap_name)
    
    for _, row in connections.iterrows():
        ax.plot(
            [row['long_orig'], row['long_dest']],
            [row['lat_orig'], row['lat_dest']],
            color=cmap(norm(row[value_col])),
            linewidth=0.5,
            transform=ccrs.Geodetic(),
        )
    
    ax.set_title(title, fontsize=10)
    
    # Add a colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label('Passenger Volume', fontsize=8)
    cbar.ax.tick_params(labelsize=6)


def main():
    print("Starting analysis and plot generation...")
    
    # --- Load Data ---
    data_hist = pd.read_csv('../data/data_with_features.csv')
    data_pred = pd.read_csv('../predictions/future_predictions_2020_2030.csv')
    data_full = pd.concat([data_hist, data_pred], ignore_index=True)
    
    # =================================================================
    # PLOT 1: Time-Series for Key Routes
    # =================================================================
    routes = [
        ('India', 'South Africa'),
        ('Brazil', 'Nigeria'),
        ('Indonesia', 'Ethiopia'),
        ('Malaysia', 'Madagascar'),
        ('Colombia', 'Algeria'),
        ('Vietnam', 'Botswana')
    ]
    
    # Call the corrected function
    plot_timeseries_for_routes(data_full, routes, '../plots/key_routes_timeseries_grid.png')

    # =================================================================
    # PLOT 2 & 3: Individual Heatmaps for Each Model (Saved as PDFs)
    # =================================================================
    print("Generating model comparison heatmaps for Dec 2019...")
    try:
        # Load prediction files
        data_xgb_pred = pd.read_csv('../predictions/prediction_xgbooost.csv')
        data_trans_pred = pd.read_csv('../predictions/prediction_transform.csv')

        # Calculate absolute error for both models
        data_xgb_pred['Absolute_Error'] = abs(data_xgb_pred['Reported.Est..Pax'] - data_xgb_pred['prediction'])
        data_trans_pred['Absolute_Error'] = abs(data_trans_pred['Reported.Est..Pax'] - data_trans_pred['prediction'])

        # Create pivot tables for the heatmaps
        error_xgb_pivot = data_xgb_pred.groupby(['Orig.Country', 'Dest.Country'])['Absolute_Error'].mean().unstack()
        error_trans_pivot = data_trans_pred.groupby(['Orig.Country', 'Dest.Country'])['Absolute_Error'].mean().unstack()

        # --- Generate and Save XGBoost Heatmap ---
        print("-> Generating XGBoost MAE heatmap...")
        fig_xgb, ax_xgb = plt.subplots(figsize=(25, 20))
        sns.heatmap(error_xgb_pivot, ax=ax_xgb, cmap="YlGnBu", linewidths=.5, annot=False)
        ax_xgb.set_title('Mean Absolute Error Heatmap - XGBoost (December 2019)', fontsize=16)
        ax_xgb.set_xlabel('Destination Country', fontsize=14)
        ax_xgb.set_ylabel('Origin Country', fontsize=14)
        plt.tight_layout()
        plt.savefig('../plots/heatmap_mae_xgboost.pdf', format='pdf')
        plt.close(fig_xgb)
        print("   Saved to ../plots/heatmap_mae_xgboost.pdf")

        # --- Generate and Save Transformer Heatmap ---
        print("-> Generating Transformer MAE heatmap...")
        fig_trans, ax_trans = plt.subplots(figsize=(25, 20))
        sns.heatmap(error_trans_pivot, ax=ax_trans, cmap="YlGnBu", linewidths=.5, annot=False)
        ax_trans.set_title('Mean Absolute Error Heatmap - Transformer (December 2019)', fontsize=16)
        ax_trans.set_xlabel('Destination Country', fontsize=14)
        ax_trans.set_ylabel('Origin Country', fontsize=14)
        plt.tight_layout()
        plt.savefig('../plots/heatmap_mae_transformer.pdf', format='pdf')
        plt.close(fig_trans)
        print("   Saved to ../plots/heatmap_mae_transformer.pdf")

    except FileNotFoundError:
        print("\nSkipping error heatmaps: Required prediction files not found.")
        print("Please ensure 'prediction_xgbooost.csv' and 'prediction_transform.csv' are in the 'predictions/' folder.\n")

    # =================================================================
    # PLOT 4 & 5: Individual Global Air Flow Maps
    # =================================================================
    print("Generating global air flow maps...")
    iata = pd.read_csv('../data/iata-icao.csv')
    
    def prepare_map_data(df, year, iata_df):
        df_year = df[df['year'] == year].copy()
        df_year = df_year.merge(iata_df[['iata', 'latitude', 'longitude']], left_on='Orig', right_on='iata', how='left').rename(columns={'latitude': 'lat_orig', 'longitude': 'long_orig'})
        df_year = df_year.merge(iata_df[['iata', 'latitude', 'longitude']], left_on='Dest', right_on='iata', how='left').rename(columns={'latitude': 'lat_dest', 'longitude': 'long_dest'})
        return df_year

    data_2019 = prepare_map_data(data_full, 2019, iata)
    data_2030 = prepare_map_data(data_full, 2030, iata)
    
    # --- Generate and Save 2019 Map ---
    print("-> Generating 2019 Global Air Flow map...")
    fig_2019 = plt.figure(figsize=(15, 8))
    ax_2019 = fig_2019.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    plot_connection_map(ax_2019, data_2019, 'Reported.Est..Pax', 'viridis', 'Global Air Flow - 2019')
    plt.savefig('../plots/global_flow_2019.png', dpi=300, bbox_inches='tight')
    plt.close(fig_2019)
    print("   Saved to ../plots/global_flow_2019.pdf")

    # --- Generate and Save 2030 Map ---
    print("-> Generating 2030 Global Air Flow map...")
    fig_2030 = plt.figure(figsize=(15, 8))
    ax_2030 = fig_2030.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    plot_connection_map(ax_2030, data_2030, 'Reported.Est..Pax', 'plasma', 'Predicted Global Air Flow - 2030')
    plt.savefig('../plots/global_flow_2030.png', dpi=300, bbox_inches='tight')
    plt.close(fig_2030)
    print("   Saved to ../plots/global_flow_2030.pdf")


if __name__ == '__main__':
    main()