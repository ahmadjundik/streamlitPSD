import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Import numpy here
import plotly.graph_objects as go
import plotly.express as px
    
st.title("Bank Customer Churn Visualization")

#@st.cache(allow_output_mutation=True)
@st.cache_data
def load_data():
    data = pd.read_csv('BankChurnersIni.csv', delimiter=';')
    data['Churn'] = data['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)
    return data

df = load_data()

attribute = st.sidebar.selectbox("Choose an attribute for analysis:", ("Gender", "Age", "Education Level", "Marital Status", "Income Category", "Card Category","Months on Book","Total Relationship Count", "Months Inactive 12 Mon","Contacts Count 12 mon","Credit Limit","Total Revolving Balance","Total Transaction Amount","Total Transaction Count" ))

if attribute == 'Gender':
    if 'Gender' in df.columns:
        gender = st.sidebar.multiselect('Select Gender:', options=df['Gender'].unique(), default=df['Gender'].unique())
        df_filtered = df[df['Gender'].isin(gender)]
        st.title("Dashboard Customer Churn by Gender")
        
        gender_counts = df_filtered['Gender'].value_counts().reindex(['M', 'F']).fillna(0).astype(int)
        churn_counts = df_filtered[df_filtered['Churn'] == 1]['Gender'].value_counts().reindex(['M', 'F']).fillna(0).astype(int)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Male", value=gender_counts.get('M', 0))
        with col2:
            st.metric(label="Female", value=gender_counts.get('F', 0))
        with col3:
            st.metric(label="Total Churn", value=churn_counts.sum())

        st.subheader("Customer Churn by Gender")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=gender_counts.index, y=gender_counts.values, color='lightblue', label='Total')
        sns.barplot(x=churn_counts.index, y=churn_counts.values, color='darkblue', label='Churn')
        
        plt.title('Customer Churn by Gender')
        plt.ylabel('Count')
        plt.legend()
        st.pyplot(fig)

elif attribute == 'Age':
    if 'Customer_Age' in df.columns:
        # Update Attrition_Flag to simpler 'Churn'/'Non-Churn' labels
        df['Churn Status'] = df['Attrition_Flag'].replace({
            'Existing Customer': 'Non-Churn',
            'Attrited Customer': 'Churn'
        })

        # Slider for age selection
        age_selection = st.sidebar.slider('Select Age Range:', min_value=int(df['Customer_Age'].min()), max_value=int(df['Customer_Age'].max()), value=(int(df['Customer_Age'].min()), int(df['Customer_Age'].max())))
        df_filtered = df[(df['Customer_Age'] >= age_selection[0]) & (df['Customer_Age'] <= age_selection[1])]
        st.title("Dashboard Customer Churn by Age")
        
        # Interactive histogram using Plotly
        hist_fig = px.histogram(df_filtered, x='Customer_Age', color='Churn Status',
                                labels={'Customer_Age': 'Age', 'Churn Status': 'Churn Status'},
                                title='Customer Churn by Age',
                                barmode='stack',
                                nbins=int(df['Customer_Age'].max() - df['Customer_Age'].min())  # Adjust bin size based on data
                               )
        hist_fig.update_layout(xaxis_title='Age', yaxis_title='Count of Customers',
                               legend_title="Churn Status")
        st.plotly_chart(hist_fig, use_container_width=True)


elif attribute == 'Education Level':
    if 'Education_Level' in df.columns:
        education_levels = st.sidebar.multiselect('Select Education Level:', options=df['Education_Level'].unique(), default=df['Education_Level'].unique())
        df_filtered = df[df['Education_Level'].isin(education_levels)]
        st.title("Dashboard Customer Churn by Education Level")
        
        education_counts = df_filtered['Education_Level'].value_counts().fillna(0).astype(int)
        churn_counts = df_filtered[df_filtered['Churn'] == 1]['Education_Level'].value_counts().fillna(0).astype(int)

        st.subheader("Customer Churn by Education Level")
        fig, ax = plt.subplots(figsize=(8, 4))
        non_churn_color = '#B0C4DE'
        churn_color = '#DC143C'
        sns.barplot(x=education_counts.index, y=education_counts.values, color=non_churn_color, label='Non-Churn')
        sns.barplot(x=churn_counts.index, y=churn_counts.values, color=churn_color, label='Churn')
        
        for p, label in zip(ax.patches[len(education_counts):], churn_counts.values):
            ax.annotate(label, (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

        plt.title('Customer Churn by Education Level')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)

elif attribute == 'Marital Status':
    if 'Marital_Status' in df.columns:
        marital_statuses = df['Marital_Status'].unique()
        df['Churn Status'] = df['Churn'].map({0: 'Non-Churn', 1: 'Churn'})
        st.title("Dashboard Customer Churn by Marital Status")

        # Prepare data for Sankey diagram
        source = []
        target = []
        value = []
        label = list(marital_statuses) + ['Non-Churn', 'Churn']

        for i, status in enumerate(marital_statuses):
            non_churn_count = df[(df['Marital_Status'] == status) & (df['Churn'] == 0)].shape[0]
            churn_count = df[(df['Marital_Status'] == status) & (df['Churn'] == 1)].shape[0]

            source.extend([i, i])
            target.extend([len(marital_statuses), len(marital_statuses) + 1])
            value.extend([non_churn_count, churn_count])

        # Sankey plot
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=label,
                color="blue"
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=["rgba(0,176,246,0.5)", "rgba(246,78,139,0.6)"] * int(len(source)/2)
            ))])

        fig.update_layout(title_text="Customer Churn Distribution by Marital Status", font_size=10)
        st.plotly_chart(fig)


elif attribute == 'Income Category':
    if 'Income_Category' in df.columns:
        income_categories = st.sidebar.multiselect('Select Income Category:', options=df['Income_Category'].unique(), default=df['Income_Category'].unique())
        df_filtered = df[df['Income_Category'].isin(income_categories)]
        st.title("Dashboard Customer Churn by Income Category")
        
        # Pie chart for overall distribution
        pie_fig = px.pie(df_filtered, names='Income_Category', title='Distribution of Income Categories')
        st.plotly_chart(pie_fig, use_container_width=True)
        
        # Bar chart for churn analysis
        churn_data = df_filtered.groupby('Income_Category')['Churn'].value_counts().unstack().fillna(0)
        churn_data.columns = ['Non-Churn', 'Churn']
        bar_fig = px.bar(churn_data, barmode='group', title="Customer Churn by Income Category")
        st.plotly_chart(bar_fig, use_container_width=True)

elif attribute == 'Card Category':
    if 'Card_Category' in df.columns:
        card_categories = st.sidebar.multiselect('Select Card Category:', options=df['Card_Category'].unique(), default=df['Card_Category'].unique())
        df_filtered = df[df['Card_Category'].isin(card_categories)]
        st.title("Dashboard Customer Churn by Card Category")
        
        # Pie chart for overall distribution of card categories
        pie_fig = px.pie(df_filtered, names='Card_Category', title='Distribution of Card Categories')
        st.plotly_chart(pie_fig, use_container_width=True)
        
        # Bar chart for churn analysis by card category
        churn_data = df_filtered.groupby('Card_Category')['Churn'].value_counts().unstack().fillna(0)
        churn_data.columns = ['Non-Churn', 'Churn']
        bar_fig = px.bar(churn_data, barmode='group', title="Customer Churn by Card Category",
                         labels={'value':'Number of Customers', 'variable':'Churn Status'},
                         log_y=True,  # Set log scale for the y-axis
                         text_auto='.2s',  # Auto-format the text labels with precision
                         )
        st.plotly_chart(bar_fig, use_container_width=True)

elif attribute == 'Months on Book':
    if 'Months_on_book' in df.columns:
        # Update Attrition_Flag to simpler 'Churn'/'Non-Churn' labels
        df['Churn Status'] = df['Attrition_Flag'].replace({
            'Existing Customer': 'Non-Churn',
            'Attrited Customer': 'Churn'
        })

        # Filter data based on sidebar selection
        months_selection = st.sidebar.slider('Select Range of Months:', min_value=int(df['Months_on_book'].min()), max_value=int(df['Months_on_book'].max()), value=(int(df['Months_on_book'].min()), int(df['Months_on_book'].max())))
        df_filtered = df[(df['Months_on_book'] >= months_selection[0]) & (df['Months_on_book'] <= months_selection[1])]
        st.title("Dashboard Customer Churn by Months on Book")
        
        # Histogram for distribution of months on book
        hist_fig = px.histogram(df_filtered, x='Months_on_book', color='Churn Status',
                                labels={'Months_on_book': 'Months on Book', 'Churn Status': 'Churn Status'},
                                title='Histogram of Months on Book')
        st.plotly_chart(hist_fig, use_container_width=True)

        # Box plot for months on book
        box_fig = px.box(df_filtered, y='Months_on_book', color='Churn Status',
                         labels={'Months_on_book': 'Months on Book', 'Churn Status': 'Churn Status'},
                         title='Box Plot of Months on Book')
        st.plotly_chart(box_fig, use_container_width=True)



elif attribute == 'Total Relationship Count':
    if 'Total_Relationship_Count' in df.columns:
        # Allow selection of specific product counts
        product_counts = st.sidebar.multiselect('Select Number of Products:', options=df['Total_Relationship_Count'].unique(), default=df['Total_Relationship_Count'].unique())
        df_filtered = df[df['Total_Relationship_Count'].isin(product_counts)]
        st.title("Dashboard Customer Churn by Total Relationship Count")
        
        # Count of customers by number of products
        product_distribution = df_filtered['Total_Relationship_Count'].value_counts().sort_index()
        dist_fig = px.bar(product_distribution, x=product_distribution.index, y=product_distribution.values,
                          labels={'x': 'Total Number of Products', 'y': 'Number of Customers'},
                          title="Distribution of Total Number of Products")
        st.plotly_chart(dist_fig, use_container_width=True)
        
        # Churn analysis by number of products
        churn_data = df_filtered.groupby('Total_Relationship_Count')['Churn'].value_counts().unstack().fillna(0)
        churn_data.columns = ['Non-Churn', 'Churn']
        churn_fig = px.bar(churn_data, barmode='group', title="Customer Churn by Total Number of Products",
                           labels={'value': 'Number of Customers', 'variable': 'Customer Status'})
        st.plotly_chart(churn_fig, use_container_width=True)

elif attribute == 'Months Inactive 12 Mon':
    if 'Months_Inactive_12_mon' in df.columns:
        # Allow selection of specific inactive months
        inactive_months = st.sidebar.multiselect('Select Inactive Months:', options=sorted(df['Months_Inactive_12_mon'].unique()), default=sorted(df['Months_Inactive_12_mon'].unique()))
        df_filtered = df[df['Months_Inactive_12_mon'].isin(inactive_months)]
        st.title("Dashboard Customer Churn by Months Inactive")
        

        # Count of customers by number of inactive months
        inactive_distribution = df_filtered['Months_Inactive_12_mon'].value_counts().sort_index()
        dist_fig = px.bar(inactive_distribution, x=inactive_distribution.index, y=inactive_distribution.values,
                          labels={'x': 'Months Inactive', 'y': 'Number of Customers'},
                          title="Distribution of Months Inactive")
        st.plotly_chart(dist_fig, use_container_width=True)
        
        # Churn analysis by number of inactive months
        churn_data = df_filtered.groupby('Months_Inactive_12_mon')['Churn'].value_counts().unstack().fillna(0)
        churn_data.columns = ['Non-Churn', 'Churn']
        churn_fig = px.bar(churn_data, barmode='group', title="Customer Churn by Months Inactive",
                           labels={'value': 'Number of Customers', 'variable': 'Customer Status'},
                           color_discrete_map={'Non-Churn': 'blue', 'Churn': 'red'})
        st.plotly_chart(churn_fig, use_container_width=True)



elif attribute == 'Contacts Count 12 mon':
    if 'Contacts_Count_12_mon' in df.columns:
        # Update Attrition_Flag to simpler 'Churn'/'Non-Churn' labels
        df['Churn Status'] = df['Attrition_Flag'].replace({
            'Existing Customer': 'Non-Churn',
            'Attrited Customer': 'Churn'
        })

        # Sidebar multiselect for choosing contact counts
        selected_contacts = st.sidebar.multiselect(
            "Select Contact Counts:",
            options=df['Contacts_Count_12_mon'].unique(),
            default=df['Contacts_Count_12_mon'].unique()
        )

        df_filtered = df[df['Contacts_Count_12_mon'].isin(selected_contacts)]

        st.title("Dashboard Customer Churn by Contacts Count 12 Mon")  # Title added here

        # Prepare data for plotting
        total_contacts = df_filtered['Contacts_Count_12_mon'].value_counts().sort_index()
        churn_data = df_filtered[df_filtered['Churn Status'] == 'Churn']['Contacts_Count_12_mon'].value_counts().reindex(total_contacts.index, fill_value=0)
        non_churn_data = df_filtered[df_filtered['Churn Status'] == 'Non-Churn']['Contacts_Count_12_mon'].value_counts().reindex(total_contacts.index, fill_value=0)

        # Pie chart for overall distribution
        pie_fig = go.Figure(data=[go.Pie(labels=total_contacts.index, values=total_contacts, name='Total Contacts Count')])
        pie_fig.update_traces(hole=.4, hoverinfo="label+percent+name")
        pie_fig.update_layout(title_text="Overall Distribution of Contacts Count")

        # Stacked bar chart for churn comparison
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(x=total_contacts.index, y=non_churn_data, name='Non-Churn', marker_color='lightblue'))
        bar_fig.add_trace(go.Bar(x=total_contacts.index, y=churn_data, name='Churn', marker_color='darkblue'))
        bar_fig.update_layout(barmode='stack', xaxis_title='Contacts Count', yaxis_title='Number of Customers',
                              title='Churn and Non-Churn Comparison by Contacts Count', 
                              legend=dict(
                                  yanchor="top",
                                  y=0.99,
                                  xanchor="left",
                                  x=0.01
                              ))

        # Display the plots
        st.plotly_chart(pie_fig, use_container_width=True)
        st.plotly_chart(bar_fig, use_container_width=True)

elif attribute == 'Credit Limit':
    if 'Credit_Limit' in df.columns:
        # Update Attrition_Flag to simpler 'Churn'/'Non-Churn' labels
        df['Churn Status'] = df['Attrition_Flag'].replace({
            'Existing Customer': 'Non-Churn',
            'Attrited Customer': 'Churn'
        })

        # Slider for credit limit selection
        min_credit_limit, max_credit_limit = int(df['Credit_Limit'].min()), int(df['Credit_Limit'].max())
        credit_limit_range = st.sidebar.slider('Select Credit Limit Range:', min_value=min_credit_limit, max_value=max_credit_limit, value=(min_credit_limit, max_credit_limit))
        df_filtered = df[(df['Credit_Limit'] >= credit_limit_range[0]) & (df['Credit_Limit'] <= credit_limit_range[1])]
        st.title("Dashboard Customer Churn by Credit Limit")

        # Histogram for distribution of credit limits
        hist_fig = px.histogram(df_filtered, x='Credit_Limit', color='Churn Status',
                                labels={'Credit_Limit': 'Credit Limit', 'Churn Status': 'Churn Status'},
                                title='Distribution of Credit Limits',
                                barmode='overlay',
                                nbins=50)
        hist_fig.update_layout(xaxis_title='Credit Limit', yaxis_title='Number of Customers',
                               legend_title="Churn Status")
        st.plotly_chart(hist_fig, use_container_width=True)

        # Box plot for credit limits
        box_fig = px.box(df_filtered, y='Credit_Limit', color='Churn Status',
                         labels={'Credit_Limit': 'Credit Limit', 'Churn Status': 'Churn Status'},
                         title='Box Plot of Credit Limits')
        st.plotly_chart(box_fig, use_container_width=True)


elif attribute == 'Total Revolving Balance':
    if 'Total_Revolving_Bal' in df.columns:
        # Update Attrition_Flag to simpler 'Churn'/'Non-Churn' labels
        df['Churn Status'] = df['Attrition_Flag'].replace({
            'Existing Customer': 'Non-Churn',
            'Attrited Customer': 'Churn'
        })

        # Slider for total revolving balance selection
        min_balance, max_balance = int(df['Total_Revolving_Bal'].min()), int(df['Total_Revolving_Bal'].max())
        balance_range = st.sidebar.slider('Select Total Revolving Balance Range:', min_value=min_balance, max_value=max_balance, value=(min_balance, max_balance))
        df_filtered = df[(df['Total_Revolving_Bal'] >= balance_range[0]) & (df['Total_Revolving_Bal'] <= balance_range[1])]
        st.title("Dashboard Customer Churn by Total Revolving Balance")

        # Histogram for distribution of total revolving balances
        hist_fig = px.histogram(df_filtered, x='Total_Revolving_Bal', color='Churn Status',
                                labels={'Total_Revolving_Bal': 'Total Revolving Balance', 'Churn Status': 'Churn Status'},
                                title='Distribution of Total Revolving Balances',
                                barmode='overlay',
                                nbins=50,  # Adjust the number of bins for better visualization
                                color_discrete_map={'Non-Churn': 'lightblue', 'Churn': 'darkblue'})  # Set specific colors
        hist_fig.update_layout(xaxis_title='Total Revolving Balance', yaxis_title='Number of Customers',
                               legend_title="Churn Status", plot_bgcolor='white')
        st.plotly_chart(hist_fig, use_container_width=True)

        # Box plot for total revolving balances
        box_fig = px.box(df_filtered, y='Total_Revolving_Bal', color='Churn Status',
                         labels={'Total_Revolving_Bal': 'Total Revolving Balance', 'Churn Status': 'Churn Status'},
                         title='Box Plot of Total Revolving Balances',
                         color_discrete_map={'Non-Churn': 'lightblue', 'Churn': 'darkblue'})  # Set specific colors
        st.plotly_chart(box_fig, use_container_width=True)

elif attribute == 'Total Transaction Amount':
    if 'Total_Trans_Amt' in df.columns:
        # Update Attrition_Flag to simpler 'Churn'/'Non-Churn' labels
        df['Churn Status'] = df['Attrition_Flag'].replace({
            'Existing Customer': 'Non-Churn',
            'Attrited Customer': 'Churn'
        })

        # Slider for total transaction amount selection
        min_trans_amt, max_trans_amt = int(df['Total_Trans_Amt'].min()), int(df['Total_Trans_Amt'].max())
        trans_amt_range = st.sidebar.slider('Select Total Transaction Amount Range:', min_value=min_trans_amt, max_value=max_trans_amt, value=(min_trans_amt, max_trans_amt))
        df_filtered = df[(df['Total_Trans_Amt'] >= trans_amt_range[0]) & (df['Total_Trans_Amt'] <= trans_amt_range[1])]
        st.title("Dashboard Customer Churn by Total Transaction Amount")

        # Histogram for distribution of total transaction amounts
        hist_fig = px.histogram(df_filtered, x='Total_Trans_Amt', color='Churn Status',
                                labels={'Total_Trans_Amt': 'Total Transaction Amount', 'Churn Status': 'Churn Status'},
                                title='Distribution of Total Transaction Amounts',
                                barmode='overlay',
                                nbins=50,  # Adjust the number of bins for better visualization
                                color_discrete_map={'Non-Churn': 'lightblue', 'Churn': 'darkblue'})
        hist_fig.update_layout(xaxis_title='Total Transaction Amount', yaxis_title='Number of Customers',
                               legend_title="Churn Status")
        st.plotly_chart(hist_fig, use_container_width=True)

        # Box plot for total transaction amounts
        box_fig = px.box(df_filtered, y='Total_Trans_Amt', color='Churn Status',
                         labels={'Total_Trans_Amt': 'Total Transaction Amount', 'Churn Status': 'Churn Status'},
                         title='Box Plot of Total Transaction Amounts',
                         color_discrete_map={'Non-Churn': 'lightblue', 'Churn': 'darkblue'})
        st.plotly_chart(box_fig, use_container_width=True)


elif attribute == 'Total Transaction Count':
    if 'Total_Trans_Ct' in df.columns:
        # Update Attrition_Flag to simpler 'Churn'/'Non-Churn' labels
        df['Churn Status'] = df['Attrition_Flag'].replace({
            'Existing Customer': 'Non-Churn',
            'Attrited Customer': 'Churn'
        })

        # Slider for total transaction count selection
        min_trans_ct, max_trans_ct = int(df['Total_Trans_Ct'].min()), int(df['Total_Trans_Ct'].max())
        trans_ct_range = st.sidebar.slider('Select Total Transaction Count Range:', min_value=min_trans_ct, max_value=max_trans_ct, value=(min_trans_ct, max_trans_ct))
        df_filtered = df[(df['Total_Trans_Ct'] >= trans_ct_range[0]) & (df['Total_Trans_Ct'] <= trans_ct_range[1])]
        st.title("Dashboard Customer Churn by Total Transaction Count")

        # Histogram for distribution of total transaction counts
        hist_fig = px.histogram(df_filtered, x='Total_Trans_Ct', color='Churn Status',
                                labels={'Total_Trans_Ct': 'Total Transaction Count', 'Churn Status': 'Churn Status'},
                                title='Distribution of Total Transaction Counts',
                                barmode='overlay',
                                nbins=40)  # Adjust the number of bins for better visualization
        hist_fig.update_layout(xaxis_title='Total Transaction Count', yaxis_title='Number of Customers',
                               legend_title="Churn Status")
        st.plotly_chart(hist_fig, use_container_width=True)

        # Box plot for total transaction counts
        box_fig = px.box(df_filtered, y='Total_Trans_Ct', color='Churn Status',
                         labels={'Total_Trans_Ct': 'Total Transaction Count', 'Churn Status': 'Churn Status'},
                         title='Box Plot of Total Transaction Counts')
        st.plotly_chart(box_fig, use_container_width=True)


