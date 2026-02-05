import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds

# Page configuration
st.set_page_config(
    page_title="Gym Operating Hours Optimizer",
    page_icon="🏋️",
    layout="wide"
)

# Title
st.title("🏋️ Gym Operating Hours Optimization")
st.markdown("**Optimize gym operating hours to minimize costs while maintaining service coverage**")

st.markdown("---")

# Sidebar for parameters
st.sidebar.header("⚙️ Model Parameters")

coverage_requirement = st.sidebar.slider(
    "Minimum Coverage Requirement (%)", 
    min_value=70, max_value=99, value=85, step=5
) / 100

min_hours = st.sidebar.slider(
    "Minimum Operating Hours", 
    min_value=6, max_value=16, value=8, step=1
)

staff_wage = st.sidebar.number_input(
    "Staff Hourly Wage ($)", 
    min_value=10, max_value=30, value=15, step=1
)

member_staff_ratio = st.sidebar.slider(
    "Member-to-Staff Ratio", 
    min_value=10, max_value=30, value=20, step=5
)

# Data setup
hours = list(range(6, 23))
n_hours = len(hours)

# Default attendance data
default_attendance = {
    6: 25, 7: 45, 8: 55, 9: 30, 10: 20, 11: 25, 12: 50, 13: 45,
    14: 35, 15: 40, 16: 60, 17: 85, 18: 95, 19: 90, 20: 70, 21: 45, 22: 20
}

# Operating costs
operating_costs = {
    6: 45, 7: 45, 8: 50, 9: 55, 10: 55, 11: 55, 12: 60, 13: 60,
    14: 65, 15: 65, 16: 70, 17: 75, 18: 75, 19: 75, 20: 70, 21: 60, 22: 55
}

# Allow user to edit attendance data
st.sidebar.markdown("---")
st.sidebar.header("📊 Edit Attendance Data")
use_custom = st.sidebar.checkbox("Customize attendance data")

if use_custom:
    attendance_data = {}
    for h in hours:
        time_str = f"{h}:00" if h < 12 else (f"12:00 PM" if h == 12 else f"{h-12}:00 PM")
        attendance_data[h] = st.sidebar.number_input(
            f"{time_str}", min_value=0, max_value=200, value=default_attendance[h], key=f"att_{h}"
        )
else:
    attendance_data = default_attendance

attendance = np.array([attendance_data[h] for h in hours])
op_costs = np.array([operating_costs[h] for h in hours])

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Predicted Attendance Pattern")
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(range(n_hours), attendance, color='steelblue', edgecolor='navy', alpha=0.7)
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Predicted Attendance')
    ax1.set_xticks(range(n_hours))
    ax1.set_xticklabels([f'{h}:00' for h in hours], rotation=45)
    ax1.axhline(y=np.mean(attendance), color='red', linestyle='--', label=f'Average: {np.mean(attendance):.1f}')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.subheader("📋 Attendance Data")
    df_attendance = pd.DataFrame({
        'Hour': hours,
        'Time': [f"{h}:00 AM" if h < 12 else (f"12:00 PM" if h == 12 else f"{h-12}:00 PM") for h in hours],
        'Attendance': attendance,
        'Op Cost ($)': op_costs
    })
    st.dataframe(df_attendance, use_container_width=True, height=400)

st.markdown("---")

# Run optimization
if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
    
    with st.spinner("Solving optimization model..."):
        # Model setup
        total_attendance = np.sum(attendance)
        min_staff_base = 2
        big_M = 10
        n_vars = 2 * n_hours
        
        min_staff_needed = np.array([
            max(min_staff_base, int(np.ceil(attendance_data[h] / member_staff_ratio))) 
            for h in hours
        ])
        
        # Objective coefficients
        c = np.concatenate([op_costs, np.full(n_hours, staff_wage)])
        
        # Build constraints
        A_ub_list = []
        b_ub_list = []
        
        # Constraint 1: Attendance Coverage
        attendance_row = np.zeros(n_vars)
        attendance_row[:n_hours] = -attendance
        A_ub_list.append(attendance_row)
        b_ub_list.append(-coverage_requirement * total_attendance)
        
        # Constraint 2: Minimum Operating Hours
        min_hours_row = np.zeros(n_vars)
        min_hours_row[:n_hours] = -1
        A_ub_list.append(min_hours_row)
        b_ub_list.append(-min_hours)
        
        # Constraint 3: Contiguity
        for i in range(1, n_hours - 1):
            contiguity_row = np.zeros(n_vars)
            contiguity_row[i-1] = 1
            contiguity_row[i] = -1
            contiguity_row[i+1] = 1
            A_ub_list.append(contiguity_row)
            b_ub_list.append(1)
        
        # Constraint 4: Minimum Staffing
        for i in range(n_hours):
            staff_req_row = np.zeros(n_vars)
            staff_req_row[i] = min_staff_needed[i]
            staff_req_row[n_hours + i] = -1
            A_ub_list.append(staff_req_row)
            b_ub_list.append(0)
        
        # Constraint 5: No Staff When Closed
        for i in range(n_hours):
            no_staff_row = np.zeros(n_vars)
            no_staff_row[i] = -big_M
            no_staff_row[n_hours + i] = 1
            A_ub_list.append(no_staff_row)
            b_ub_list.append(0)
        
        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
        
        # Bounds and integrality
        lb = np.zeros(n_vars)
        ub = np.concatenate([np.ones(n_hours), np.full(n_hours, big_M)])
        bounds = Bounds(lb, ub)
        integrality = np.ones(n_vars, dtype=int)
        constraints = LinearConstraint(A_ub, -np.inf, b_ub)
        
        # Solve
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if result.success:
        st.success("✅ Optimal solution found!")
        
        # Extract solution
        x_solution = np.round(result.x[:n_hours]).astype(int)
        s_solution = np.round(result.x[n_hours:]).astype(int)
        
        # Calculate results
        open_hours_list = [hours[i] for i in range(n_hours) if x_solution[i] == 1]
        total_cost = result.fun
        total_attendance_served = sum(attendance[i] for i in range(n_hours) if x_solution[i] == 1)
        full_op_cost = sum(op_costs[i] + min_staff_needed[i] * staff_wage for i in range(n_hours))
        savings = full_op_cost - total_cost
        
        # Display summary metrics
        st.markdown("### 📊 Results Summary")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            opening = min(open_hours_list)
            closing = max(open_hours_list) + 1
            opening_str = f"{opening}:00 AM" if opening < 12 else f"{opening-12}:00 PM"
            closing_str = f"{closing-12}:00 PM" if closing > 12 else f"{closing}:00"
            st.metric("🕐 Operating Hours", f"{opening_str} - {closing_str}")
        
        with metric_col2:
            st.metric("⏱️ Total Hours", f"{len(open_hours_list)} hours", f"-{17 - len(open_hours_list)} from full")
        
        with metric_col3:
            st.metric("💰 Daily Cost", f"${total_cost:,.0f}", f"-${savings:,.0f} ({savings/full_op_cost*100:.1f}%)")
        
        with metric_col4:
            st.metric("👥 Coverage", f"{total_attendance_served/total_attendance*100:.1f}%", f"{total_attendance_served} members")
        
        # Annual savings highlight
        st.markdown(f"""
        ### 💵 Annual Savings: **${savings * 365:,.0f}**
        """)
        
        # Results visualization
        st.markdown("### 📈 Optimization Results")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            # Operating Schedule
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            colors = ['green' if h in open_hours_list else 'red' for h in hours]
            ax2.bar(range(n_hours), attendance, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Attendance')
            ax2.set_title('Optimal Operating Schedule\n(Green = Open, Red = Closed)')
            ax2.set_xticks(range(n_hours))
            ax2.set_xticklabels([f'{h}:00' for h in hours], rotation=45)
            plt.tight_layout()
            st.pyplot(fig2)
        
        with res_col2:
            # Cost Comparison
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            categories = ['Full Operation\n(17 hours)', f'Optimized\n({len(open_hours_list)} hours)']
            costs = [full_op_cost, total_cost]
            bars = ax3.bar(categories, costs, color=['coral', 'steelblue'], alpha=0.7, edgecolor='black')
            ax3.set_ylabel('Daily Cost ($)')
            ax3.set_title('Cost Comparison')
            for bar, cost in zip(bars, costs):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                        f'${cost:.0f}', ha='center', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig3)
        
        # Detailed schedule table
        st.markdown("### 📋 Detailed Schedule")
        
        results_data = []
        for i in range(n_hours):
            h = hours[i]
            is_open = x_solution[i] == 1
            staff = s_solution[i]
            time_str = f"{h}:00 AM" if h < 12 else (f"12:00 PM" if h == 12 else f"{h-12}:00 PM")
            
            if is_open:
                hour_total = op_costs[i] + staff * staff_wage
                results_data.append({
                    'Hour': h,
                    'Time': time_str,
                    'Status': '✅ OPEN',
                    'Attendance': attendance[i],
                    'Staff': staff,
                    'Op Cost': f"${op_costs[i]}",
                    'Staff Cost': f"${staff * staff_wage}",
                    'Total': f"${hour_total}"
                })
            else:
                results_data.append({
                    'Hour': h,
                    'Time': time_str,
                    'Status': '❌ CLOSED',
                    'Attendance': '-',
                    'Staff': '-',
                    'Op Cost': '-',
                    'Staff Cost': '-',
                    'Total': '-'
                })
        
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True)
        
        # Staffing chart
        st.markdown("### 👥 Staffing Schedule")
        fig4, ax4 = plt.subplots(figsize=(12, 4))
        ax4.bar(range(n_hours), s_solution, color='purple', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Number of Staff')
        ax4.set_title('Optimal Staffing Schedule')
        ax4.set_xticks(range(n_hours))
        ax4.set_xticklabels([f'{h}:00' for h in hours], rotation=45)
        plt.tight_layout()
        st.pyplot(fig4)
        
    else:
        st.error(f"❌ Optimization failed: {result.message}")
        st.info("Try adjusting the parameters (lower coverage requirement or fewer minimum hours)")

# Footer
st.markdown("---")
st.markdown("""
**About this app:** This optimization tool uses Mixed Integer Linear Programming (MILP) to find 
the optimal gym operating hours that minimize costs while meeting service coverage requirements.

Built with Streamlit, SciPy, and Python.
""")
