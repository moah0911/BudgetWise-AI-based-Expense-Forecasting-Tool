"""
BudgetWise Forecasting - Streamlit Dashboard
Interactive web application for personal expense forecasting and budget optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="BudgetWise Forecasting",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_config():
    """Load configuration."""
    try:
        with open("config/config.yaml", 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error("Configuration file not found. Please ensure config/config.yaml exists.")
        return {}

@st.cache_data
def load_data():
    """Load processed and feature data."""
    try:
        # Load processed data
        processed_path = Path("data/processed/")
        processed_files = list(processed_path.glob("*.csv"))
        
        if processed_files:
            data = pd.read_csv(processed_files[0])
            data['date'] = pd.to_datetime(data['date'])
            return data
        else:
            st.error("No processed data found. Please run data preprocessing first.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_model_performance():
    """Load model performance data."""
    try:
        models_path = Path("models")
        performance_files = {
            'baseline': 'baseline_performance.pkl',
            'ml': 'ml_performance.pkl', 
            'dl': 'dl_performance.pkl'
        }
        
        performance_data = {}
        for model_type, filename in performance_files.items():
            file_path = models_path / filename
            if file_path.exists():
                performance_data[model_type] = joblib.load(file_path)
        
        return performance_data
    except Exception as e:
        st.error(f"Error loading model performance: {str(e)}")
        return {}

def create_expense_overview(data):
    """Create expense overview visualizations."""
    st.subheader("📊 Expense Overview")
    
    if data.empty:
        st.warning("No data available for visualization.")
        return
    
    # Daily expense trend
    daily_expenses = data.groupby('date')['amount'].sum().reset_index()
    
    fig = px.line(
        daily_expenses, 
        x='date', 
        y='amount',
        title="Daily Expense Trend",
        labels={'amount': 'Amount ($)', 'date': 'Date'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Expense by category
    col1, col2 = st.columns(2)
    
    with col1:
        if 'category' in data.columns:
            category_expenses = data.groupby('category')['amount'].sum().reset_index()
            category_expenses = category_expenses.sort_values('amount', ascending=False).head(10)
            
            fig = px.bar(
                category_expenses,
                x='amount',
                y='category',
                orientation='h',
                title="Top 10 Expense Categories",
                labels={'amount': 'Total Amount ($)', 'category': 'Category'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'category' in data.columns:
            # Pie chart for category distribution
            fig = px.pie(
                category_expenses.head(8),
                values='amount',
                names='category',
                title="Expense Distribution by Category"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def create_model_performance_dashboard(performance_data):
    """Create model performance dashboard."""
    st.subheader("🎯 Model Performance Dashboard")
    
    if not performance_data:
        st.warning("No model performance data available. Please train models first.")
        return
    
    # Combine all performance data
    all_models = {}
    for model_type, categories in performance_data.items():
        for category, models in categories.items():
            for model_name, metrics in models.items():
                key = f"{model_type}_{model_name}_{category}"
                all_models[key] = {**metrics, 'model_type': model_type, 'category': category, 'model_name': model_name}
    
    if not all_models:
        st.warning("No model performance metrics available.")
        return
    
    # Convert to DataFrame
    performance_df = pd.DataFrame.from_dict(all_models, orient='index')
    
    # Model type performance comparison
    col1, col2 = st.columns(2)
    
    with col1:
        if 'MAE' in performance_df.columns:
            avg_mae_by_type = performance_df.groupby('model_type')['MAE'].mean().reset_index()
            
            fig = px.bar(
                avg_mae_by_type,
                x='model_type',
                y='MAE',
                title="Average MAE by Model Type",
                labels={'MAE': 'Mean Absolute Error', 'model_type': 'Model Type'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Directional_Accuracy' in performance_df.columns:
            avg_accuracy_by_type = performance_df.groupby('model_type')['Directional_Accuracy'].mean().reset_index()
            
            fig = px.bar(
                avg_accuracy_by_type,
                x='model_type',
                y='Directional_Accuracy',
                title="Average Directional Accuracy by Model Type",
                labels={'Directional_Accuracy': 'Directional Accuracy (%)', 'model_type': 'Model Type'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Best models per category
    st.subheader("🏆 Best Models per Category")
    
    best_models = {}
    for category in performance_df['category'].unique():
        category_data = performance_df[performance_df['category'] == category]
        if 'MAE' in category_data.columns:
            best_model_idx = category_data['MAE'].idxmin()
            best_models[category] = category_data.loc[best_model_idx]
    
    if best_models:
        best_models_df = pd.DataFrame.from_dict(best_models, orient='index')
        best_models_df = best_models_df.reset_index().rename(columns={'index': 'Category'})
        
        # Display as table
        display_columns = ['Category', 'model_type', 'model_name', 'MAE', 'RMSE', 'Directional_Accuracy']
        available_columns = [col for col in display_columns if col in best_models_df.columns]
        
        st.dataframe(
            best_models_df[available_columns].round(3),
            use_container_width=True
        )

def create_forecasting_interface(data):
    """Create forecasting interface."""
    st.subheader("🔮 Expense Forecasting")
    
    if data.empty:
        st.warning("No data available for forecasting.")
        return
    
    st.info("📋 **Note**: This is a demo interface. In a production environment, trained models would be loaded and used for real-time forecasting.")
    
    # Forecasting parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_horizon = st.selectbox(
            "Forecast Horizon",
            options=[7, 14, 30, 90],
            index=2,
            help="Number of days to forecast"
        )
    
    with col2:
        if 'category' in data.columns:
            categories = ['All'] + data['category'].unique().tolist()
            selected_category = st.selectbox(
                "Expense Category",
                options=categories,
                help="Select category to forecast"
            )
    
    with col3:
        confidence_interval = st.slider(
            "Confidence Interval (%)",
            min_value=80,
            max_value=99,
            value=95,
            help="Confidence interval for predictions"
        )
    
    # Generate demo forecast
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            # Demo forecast (in production, this would use trained models)
            last_date = data['date'].max()
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_horizon,
                freq='D'
            )
            
            # Demo predictions (normally from trained models)
            if selected_category == 'All':
                recent_avg = data.groupby('date')['amount'].sum().tail(30).mean()
            else:
                category_data = data[data['category'] == selected_category] if 'category' in data.columns else data
                recent_avg = category_data.groupby('date')['amount'].sum().tail(30).mean()
            
            # Generate synthetic forecast
            np.random.seed(42)
            base_forecast = recent_avg * (1 + np.random.normal(0, 0.1, forecast_horizon))
            upper_bound = base_forecast * 1.2
            lower_bound = base_forecast * 0.8
            
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'forecast': base_forecast,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound
            })
            
            # Plot forecast
            fig = go.Figure()
            
            # Historical data
            if selected_category == 'All':
                historical = data.groupby('date')['amount'].sum().reset_index()
            else:
                historical_data = data[data['category'] == selected_category] if 'category' in data.columns else data
                historical = historical_data.groupby('date')['amount'].sum().reset_index()
            
            fig.add_trace(go.Scatter(
                x=historical['date'].tail(30),
                y=historical['amount'].tail(30),
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['upper_bound'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['lower_bound'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name=f'{confidence_interval}% Confidence Interval',
                fillcolor='rgba(255,0,0,0.2)'
            ))
            
            fig.update_layout(
                title=f"Expense Forecast - {selected_category} ({forecast_horizon} days)",
                xaxis_title="Date",
                yaxis_title="Amount ($)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast summary
            st.subheader("📊 Forecast Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Avg Daily Forecast",
                    f"${forecast_df['forecast'].mean():.2f}",
                    delta=f"{((forecast_df['forecast'].mean() / recent_avg) - 1) * 100:.1f}%"
                )
            
            with col2:
                st.metric(
                    "Total Forecast",
                    f"${forecast_df['forecast'].sum():.2f}"
                )
            
            with col3:
                st.metric(
                    "Max Daily Expense",
                    f"${forecast_df['forecast'].max():.2f}"
                )
            
            with col4:
                st.metric(
                    "Min Daily Expense",
                    f"${forecast_df['forecast'].min():.2f}"
                )

def create_budget_optimizer():
    """Create budget optimization interface."""
    st.subheader("💰 Budget Optimization")
    
    st.info("📋 **Note**: This is a demo interface. In a production environment, optimization algorithms would provide personalized budget recommendations.")
    
    # Budget parameters
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_income = st.number_input(
            "Monthly Income ($)",
            min_value=0.0,
            value=5000.0,
            step=100.0,
            help="Your total monthly income"
        )
        
        savings_goal = st.slider(
            "Savings Goal (%)",
            min_value=5,
            max_value=50,
            value=20,
            help="Percentage of income to save"
        )
    
    with col2:
        expense_categories = [
            "Housing", "Food", "Transportation", "Entertainment",
            "Healthcare", "Shopping", "Utilities", "Others"
        ]
        
        priority_category = st.selectbox(
            "Priority Category",
            options=expense_categories,
            help="Category to prioritize in budget"
        )
        
        optimization_goal = st.radio(
            "Optimization Goal",
            options=["Minimize Risk", "Maximize Savings", "Balanced"],
            help="Primary goal for budget optimization"
        )
    
    if st.button("Optimize Budget", type="primary"):
        with st.spinner("Optimizing budget..."):
            # Demo optimization (in production, this would use optimization algorithms)
            savings_amount = monthly_income * (savings_goal / 100)
            available_budget = monthly_income - savings_amount
            
            # Demo budget allocation
            if optimization_goal == "Minimize Risk":
                # Conservative allocation
                budget_allocation = {
                    "Housing": 0.35,
                    "Food": 0.15,
                    "Transportation": 0.10,
                    "Healthcare": 0.08,
                    "Utilities": 0.10,
                    "Entertainment": 0.08,
                    "Shopping": 0.09,
                    "Others": 0.05
                }
            elif optimization_goal == "Maximize Savings":
                # Aggressive savings
                budget_allocation = {
                    "Housing": 0.30,
                    "Food": 0.12,
                    "Transportation": 0.08,
                    "Healthcare": 0.08,
                    "Utilities": 0.08,
                    "Entertainment": 0.05,
                    "Shopping": 0.06,
                    "Others": 0.23
                }
            else:  # Balanced
                budget_allocation = {
                    "Housing": 0.32,
                    "Food": 0.14,
                    "Transportation": 0.09,
                    "Healthcare": 0.08,
                    "Utilities": 0.09,
                    "Entertainment": 0.12,
                    "Shopping": 0.11,
                    "Others": 0.05
                }
            
            # Calculate amounts
            budget_amounts = {cat: available_budget * pct for cat, pct in budget_allocation.items()}
            
            # Display optimization results
            st.subheader("🎯 Optimized Budget Plan")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Monthly Income", f"${monthly_income:,.2f}")
            
            with col2:
                st.metric("Savings", f"${savings_amount:,.2f}", f"{savings_goal}%")
            
            with col3:
                st.metric("Available Budget", f"${available_budget:,.2f}")
            
            with col4:
                st.metric("Priority Category", priority_category, f"${budget_amounts.get(priority_category, 0):,.2f}")
            
            # Budget breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart
                fig = px.pie(
                    values=list(budget_amounts.values()),
                    names=list(budget_amounts.keys()),
                    title="Budget Allocation by Category"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Bar chart
                budget_df = pd.DataFrame({
                    'Category': list(budget_amounts.keys()),
                    'Amount': list(budget_amounts.values()),
                    'Percentage': [v*100 for v in budget_allocation.values()]
                })
                
                fig = px.bar(
                    budget_df,
                    x='Amount',
                    y='Category',
                    orientation='h',
                    title="Budget Amounts by Category",
                    text='Percentage'
                )
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Budget Recommendations")
            
            recommendations = [
                f"💰 Save ${savings_amount:,.2f} ({savings_goal}%) of your monthly income",
                f"🏠 Allocate ${budget_amounts['Housing']:,.2f} for housing expenses (recommended: 25-35%)",
                f"🍽️ Budget ${budget_amounts['Food']:,.2f} for food expenses",
                f"🎯 Focus on {priority_category} with ${budget_amounts.get(priority_category, 0):,.2f} allocation",
                f"📊 Review and adjust budget monthly based on actual spending patterns"
            ]
            
            for rec in recommendations:
                st.markdown(f"- {rec}")

def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🏦 BudgetWise Forecasting</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Personal Expense Forecasting & Budget Optimization")
    
    # Load data
    config = load_config()
    data = load_data()
    performance_data = load_model_performance()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Dashboard", "📊 Expense Analysis", "🔮 Forecasting", "💰 Budget Optimizer", "🎯 Model Performance"]
    )
    
    # Main content based on selected page
    if page == "🏠 Dashboard":
        st.subheader("📋 Dashboard Overview")
        
        if not data.empty:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_expenses = data['amount'].sum()
                st.metric("Total Expenses", f"${total_expenses:,.2f}")
            
            with col2:
                avg_daily = data.groupby('date')['amount'].sum().mean()
                st.metric("Avg Daily Expense", f"${avg_daily:.2f}")
            
            with col3:
                num_transactions = len(data)
                st.metric("Total Transactions", f"{num_transactions:,}")
            
            with col4:
                if 'category' in data.columns:
                    num_categories = data['category'].nunique()
                    st.metric("Expense Categories", num_categories)
            
            # Quick insights
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.markdown("**💡 Quick Insights:**")
            
            if 'category' in data.columns:
                top_category = data.groupby('category')['amount'].sum().idxmax()
                top_amount = data.groupby('category')['amount'].sum().max()
                st.markdown(f"• Your highest expense category is **{top_category}** with ${top_amount:,.2f}")
            
            recent_trend = data.groupby('date')['amount'].sum().tail(7).mean()
            overall_avg = data.groupby('date')['amount'].sum().mean()
            trend_change = ((recent_trend / overall_avg) - 1) * 100
            
            if trend_change > 5:
                st.markdown(f"• Your recent spending is **{trend_change:.1f}% higher** than average")
            elif trend_change < -5:
                st.markdown(f"• Your recent spending is **{abs(trend_change):.1f}% lower** than average")
            else:
                st.markdown("• Your recent spending is consistent with your average")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent activity
        if not data.empty:
            st.subheader("📈 Recent Activity")
            recent_data = data.sort_values('date', ascending=False).head(10)
            st.dataframe(recent_data, use_container_width=True)
    
    elif page == "📊 Expense Analysis":
        create_expense_overview(data)
    
    elif page == "🔮 Forecasting":
        create_forecasting_interface(data)
    
    elif page == "💰 Budget Optimizer":
        create_budget_optimizer()
    
    elif page == "🎯 Model Performance":
        create_model_performance_dashboard(performance_data)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🏦 BudgetWise Forecasting**")
    st.sidebar.markdown("AI-powered financial insights")
    
    # Instructions
    with st.sidebar.expander("ℹ️ How to Use"):
        st.markdown("""
        1. **Dashboard**: Overview of your expenses
        2. **Expense Analysis**: Detailed expense visualizations
        3. **Forecasting**: Predict future expenses
        4. **Budget Optimizer**: Get budget recommendations
        5. **Model Performance**: View AI model results
        
        **Note**: Make sure to run data preprocessing and model training first!
        """)

if __name__ == "__main__":
    main()