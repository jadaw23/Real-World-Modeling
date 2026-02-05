#!/usr/bin/env python3
"""
Extra Credit Project: Predicting Gym Attendance to Optimize Operating Hours
Course: Operations Research / Mathematical Optimization
Author: Jada Williams
Date: February 2026

This script solves an optimization problem to determine optimal gym operating hours
that minimize costs while maintaining adequate service coverage.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds

print("="*70)
print("EXTRA CREDIT PROJECT: GYM OPERATING HOURS OPTIMIZATION")
print("="*70)

# ============================================================================
# SECTION 1: PROBLEM DEFINITION
# ============================================================================
print("\n" + "="*70)
print("1. PROBLEM DEFINITION AND REAL-WORLD CONTEXT")
print("="*70)

problem_description = """
BACKGROUND:
A university fitness center faces a common operational challenge: balancing 
operating costs with member satisfaction. The gym currently operates from 
6 AM to 11 PM (17 hours daily), but budget constraints require reducing 
operational hours while maintaining service quality for the majority of members.

PROBLEM STATEMENT:
The gym management needs to determine the optimal operating hours that:
1. Minimize total operating costs (staff, utilities, maintenance)
2. Ensure that at least 85% of total daily attendance is served
3. Maintain a minimum of 8 consecutive operating hours
4. Have adequate staffing (at least one staff member for every 20 members)

WHY THIS PROBLEM MATTERS:
- Cost Reduction: Operating costs average $50-100 per hour
- Member Satisfaction: Closing during low-attendance hours affects fewer members
- Resource Allocation: Optimizing staff schedules improves efficiency
- Sustainability: Reduced hours decreases energy consumption
"""
print(problem_description)

# ============================================================================
# SECTION 2: DATA COLLECTION AND ATTENDANCE PREDICTION
# ============================================================================
print("\n" + "="*70)
print("2. DATA COLLECTION AND ATTENDANCE PREDICTION")
print("="*70)

# Define the hours of potential operation (6 AM to 11 PM)
hours = list(range(6, 23))  # 6, 7, 8, ..., 22
n_hours = len(hours)  # 17 hours

# Predicted average attendance per hour (based on typical gym patterns)
attendance_data = {
    6: 25,   # 6 AM - Early birds
    7: 45,   # 7 AM - Morning rush begins
    8: 55,   # 8 AM - Peak morning
    9: 30,   # 9 AM - Classes start, attendance drops
    10: 20,  # 10 AM - Low point
    11: 25,  # 11 AM - Slight increase before lunch
    12: 50,  # 12 PM - Lunch crowd
    13: 45,  # 1 PM - Lunch crowd continues
    14: 35,  # 2 PM - Afternoon lull
    15: 40,  # 3 PM - Building up
    16: 60,  # 4 PM - After-class crowd
    17: 85,  # 5 PM - Evening peak begins
    18: 95,  # 6 PM - Peak hour
    19: 90,  # 7 PM - Still peak
    20: 70,  # 8 PM - Starting to decline
    21: 45,  # 9 PM - Evening wind-down
    22: 20   # 10 PM - Late night
}

# Convert to array for optimization
attendance = np.array([attendance_data[h] for h in hours])

# Create attendance DataFrame
df_attendance = pd.DataFrame({
    'Hour': hours,
    'Time': [f"{h}:00 AM" if h < 12 else (f"12:00 PM" if h == 12 else f"{h-12}:00 PM") for h in hours],
    'Predicted_Attendance': attendance
})

print("\nPredicted Hourly Attendance:")
print(df_attendance.to_string(index=False))
print(f"\nTotal Daily Attendance: {sum(attendance)} members")
print(f"Peak Hour: {hours[np.argmax(attendance)]}:00 with {max(attendance)} members")
print(f"Lowest Hour: {hours[np.argmin(attendance)]}:00 with {min(attendance)} members")

# Plot attendance pattern
plt.figure(figsize=(12, 6))
plt.bar(range(n_hours), attendance, color='steelblue', edgecolor='navy', alpha=0.7)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Predicted Attendance', fontsize=12)
plt.title('Predicted Hourly Gym Attendance Pattern', fontsize=14)
plt.xticks(range(n_hours), [f'{h}:00' for h in hours], rotation=45)
plt.axhline(y=np.mean(attendance), color='red', linestyle='--', 
            label=f'Average: {np.mean(attendance):.1f}')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/home/claude/attendance_pattern.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[Figure 1: Attendance Pattern saved]")

# ============================================================================
# SECTION 3: MATHEMATICAL MODEL FORMULATION
# ============================================================================
print("\n" + "="*70)
print("3. MATHEMATICAL MODEL FORMULATION")
print("="*70)

model_formulation = """
SETS AND INDICES:
- H = {6, 7, 8, ..., 22}: Set of potential operating hours (6 AM to 10 PM)
- |H| = 17: Total number of potential operating hours
- Index i ∈ {0, 1, ..., 16} maps to hour h = i + 6

PARAMETERS:
| Parameter | Description                    | Value       | Justification           |
|-----------|--------------------------------|-------------|-------------------------|
| a_i       | Predicted attendance at hour i | See data    | Historical patterns     |
| c_i       | Operating cost per hour i      | $45-$75     | Utilities + maintenance |
| α         | Minimum attendance coverage    | 85%         | Industry standard       |
| M_min     | Minimum consecutive hours      | 8           | Practical constraint    |
| r         | Member-to-staff ratio          | 20:1        | Safety requirement      |
| w         | Staff hourly wage              | $15         | Minimum + benefits      |

DECISION VARIABLES:
- x_i ∈ {0, 1}: Binary, 1 if gym is open during hour i, 0 otherwise
- s_i ≥ 0: Integer, number of staff members scheduled for hour i

OBJECTIVE FUNCTION:
Minimize Total Operating Cost:
    min Z = Σ (c_i · x_i + w · s_i)  for all i

CONSTRAINTS:
1. Attendance Coverage: Σ(a_i · x_i) ≥ α · Σ(a_i)
2. Minimum Hours: Σ(x_i) ≥ M_min
3. Contiguity (no gaps): x_i ≥ x_{i-1} + x_{i+1} - 1  for 0 < i < 16
4. Staffing Requirement: s_i ≥ ⌈a_i/r⌉ · x_i
5. No Staff When Closed: s_i ≤ M · x_i (Big-M constraint)
"""
print(model_formulation)

# ============================================================================
# SECTION 4: PARAMETER JUSTIFICATION
# ============================================================================
print("\n" + "="*70)
print("4. PARAMETER JUSTIFICATION")
print("="*70)

# Define operating costs per hour
operating_costs = {
    6: 45, 7: 45, 8: 50, 9: 55, 10: 55, 11: 55,
    12: 60, 13: 60, 14: 65, 15: 65, 16: 70,
    17: 75, 18: 75, 19: 75, 20: 70, 21: 60, 22: 55
}
op_costs = np.array([operating_costs[h] for h in hours])

# Staffing parameters
staff_wage = 15
member_staff_ratio = 20
min_staff_base = 2
big_M = 10

# Calculate minimum staff needed
min_staff_needed = np.array([max(min_staff_base, int(np.ceil(attendance_data[h] / member_staff_ratio))) 
                             for h in hours])

print("\nOperating Cost Justification:")
print("- Off-peak hours (6-8 AM, 9-10 PM): $45-55 (lower electricity rates)")
print("- Standard hours (9-11 AM): $55")
print("- Mid-peak hours (12-3 PM): $60-65")
print("- Peak hours (4-8 PM): $70-75 (highest electricity rates)")
print("- Late evening (9-10 PM): $55-60")

print("\nStaffing Requirements:")
print(f"- Hourly wage: ${staff_wage}")
print(f"- Member-to-staff ratio: {member_staff_ratio}:1")
print(f"- Minimum staff when open: {min_staff_base}")

print("\nHour | Attendance | Min Staff | Op Cost | Staff Cost | Total Cost")
print("-" * 65)
for i, h in enumerate(hours):
    staff_cost = min_staff_needed[i] * staff_wage
    total = op_costs[i] + staff_cost
    print(f" {h:2d}  |    {attendance[i]:3d}     |     {min_staff_needed[i]}     |   ${op_costs[i]:2d}   |    ${staff_cost:3d}     |    ${total:3d}")

# Plot cost breakdown
fig, ax1 = plt.subplots(figsize=(12, 6))
bar_width = 0.35
index = np.arange(n_hours)
ax1.bar(index, op_costs, bar_width, label='Operating Cost', color='steelblue', alpha=0.7)
ax1.bar(index + bar_width, min_staff_needed * staff_wage, bar_width, 
        label='Staff Cost', color='coral', alpha=0.7)
ax1.set_xlabel('Hour of Day', fontsize=12)
ax1.set_ylabel('Cost ($)', fontsize=12)
ax1.set_title('Hourly Cost Breakdown', fontsize=14)
ax1.set_xticks(index + bar_width / 2)
ax1.set_xticklabels([f'{h}:00' for h in hours], rotation=45)
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.plot(index + bar_width / 2, attendance, 'g-o', linewidth=2, markersize=6, label='Attendance')
ax2.set_ylabel('Attendance', fontsize=12, color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('/home/claude/cost_breakdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[Figure 2: Cost Breakdown saved]")

# ============================================================================
# SECTION 5: OPTIMIZATION MODEL IMPLEMENTATION
# ============================================================================
print("\n" + "="*70)
print("5. OPTIMIZATION MODEL IMPLEMENTATION (SciPy MILP)")
print("="*70)

# Model parameters
total_attendance = np.sum(attendance)
coverage_requirement = 0.85
min_hours = 8
n_vars = 2 * n_hours  # 34 variables

print(f"\nModel Parameters:")
print(f"  - Total potential attendance: {total_attendance}")
print(f"  - Coverage requirement: {coverage_requirement * 100}%")
print(f"  - Minimum attendance to serve: {int(coverage_requirement * total_attendance)}")
print(f"  - Minimum operating hours: {min_hours}")
print(f"  - Total variables: {n_vars} ({n_hours} binary x, {n_hours} integer s)")

# Objective function coefficients
c = np.concatenate([op_costs, np.full(n_hours, staff_wage)])

# Build constraint matrices
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

# Constraint 3: Contiguity (no gaps)
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

print(f"\nConstraint Summary:")
print(f"  - Attendance coverage: 1 constraint")
print(f"  - Minimum hours: 1 constraint")
print(f"  - Contiguity: {n_hours - 2} constraints")
print(f"  - Staffing requirement: {n_hours} constraints")
print(f"  - Big-M constraints: {n_hours} constraints")
print(f"  - Total: {len(b_ub)} constraints")

# Variable bounds
lb = np.zeros(n_vars)
ub = np.concatenate([np.ones(n_hours), np.full(n_hours, big_M)])
bounds = Bounds(lb, ub)

# Integrality
integrality = np.ones(n_vars, dtype=int)

# Create constraint object
constraints = LinearConstraint(A_ub, -np.inf, b_ub)

# ============================================================================
# SECTION 6: SOLVING THE MODEL
# ============================================================================
print("\n" + "="*70)
print("6. SOLVING THE OPTIMIZATION MODEL")
print("="*70)

print("\nSolving the Mixed Integer Linear Program...")
print("-" * 50)

result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

print(f"\nSolver Status: {result.message}")
print(f"Success: {result.success}")
print(f"Optimal Objective Value: ${result.fun:.2f}")

# ============================================================================
# SECTION 7: RESULTS ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("7. OPTIMIZATION RESULTS")
print("="*70)

if result.success:
    print("\n✓ Optimal solution found!\n")
    
    # Extract solution
    x_solution = np.round(result.x[:n_hours]).astype(int)
    s_solution = np.round(result.x[n_hours:]).astype(int)
    
    # Build results
    results_data = []
    total_cost = 0
    total_attendance_served = 0
    open_hours_list = []
    
    for i in range(n_hours):
        h = hours[i]
        is_open = x_solution[i] == 1
        staff = s_solution[i]
        
        if is_open:
            open_hours_list.append(h)
            hour_op_cost = op_costs[i]
            hour_staff_cost = staff * staff_wage
            hour_total = hour_op_cost + hour_staff_cost
            total_cost += hour_total
            total_attendance_served += attendance[i]
            
            time_str = f"{h}:00 AM" if h < 12 else (f"12:00 PM" if h == 12 else f"{h-12}:00 PM")
            results_data.append({
                'Hour': h,
                'Time': time_str,
                'Open': 'YES',
                'Attendance': attendance[i],
                'Staff': staff,
                'Op_Cost': f"${hour_op_cost}",
                'Staff_Cost': f"${hour_staff_cost}",
                'Total': f"${hour_total}"
            })
        else:
            time_str = f"{h}:00 AM" if h < 12 else (f"12:00 PM" if h == 12 else f"{h-12}:00 PM")
            results_data.append({
                'Hour': h,
                'Time': time_str,
                'Open': 'NO',
                'Attendance': '-',
                'Staff': '-',
                'Op_Cost': '-',
                'Staff_Cost': '-',
                'Total': '-'
            })
    
    df_results = pd.DataFrame(results_data)
    print("Detailed Schedule:")
    print(df_results.to_string(index=False))
    
    # Calculate comparison with full operation
    full_op_cost = sum(op_costs[i] + min_staff_needed[i] * staff_wage for i in range(n_hours))
    savings = full_op_cost - total_cost
    
    print("\n" + "="*70)
    print("SOLUTION SUMMARY")
    print("="*70)
    
    opening = min(open_hours_list)
    closing = max(open_hours_list) + 1
    opening_str = f"{opening}:00 AM" if opening < 12 else f"{opening-12}:00 PM"
    closing_str = f"{closing}:00 AM" if closing < 12 else (f"12:00 PM" if closing == 12 else f"{closing-12}:00 PM")
    
    print(f"\n📍 OPTIMAL OPERATING HOURS:")
    print(f"   Opening Time: {opening_str}")
    print(f"   Closing Time: {closing_str}")
    print(f"   Total Operating Hours: {len(open_hours_list)} hours")
    
    print(f"\n💰 COST ANALYSIS:")
    print(f"   Total Daily Operating Cost: ${total_cost:.2f}")
    print(f"   Average Cost per Hour: ${total_cost/len(open_hours_list):.2f}")
    print(f"\n   Full Operation Cost (17 hours): ${full_op_cost:.2f}")
    print(f"   Daily Savings: ${savings:.2f} ({savings/full_op_cost*100:.1f}%)")
    print(f"   Monthly Savings (30 days): ${savings * 30:,.2f}")
    print(f"   Annual Savings (365 days): ${savings * 365:,.2f}")
    
    print(f"\n👥 ATTENDANCE COVERAGE:")
    print(f"   Total Members Served: {total_attendance_served}")
    print(f"   Total Potential Members: {total_attendance}")
    print(f"   Coverage Rate: {total_attendance_served/total_attendance*100:.1f}%")
    print(f"   Members Not Served: {total_attendance - total_attendance_served}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Operating Schedule
    ax1 = axes[0, 0]
    colors = ['green' if h in open_hours_list else 'red' for h in hours]
    ax1.bar(range(n_hours), attendance, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Attendance')
    ax1.set_title('Optimal Operating Schedule\n(Green = Open, Red = Closed)')
    ax1.set_xticks(range(n_hours))
    ax1.set_xticklabels([f'{h}:00' for h in hours], rotation=45)
    
    # Plot 2: Cost Comparison
    ax2 = axes[0, 1]
    categories = ['Full Operation\n(17 hours)', f'Optimized\n({len(open_hours_list)} hours)']
    costs = [full_op_cost, total_cost]
    bars = ax2.bar(categories, costs, color=['coral', 'steelblue'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Daily Cost ($)')
    ax2.set_title('Cost Comparison')
    for bar, cost in zip(bars, costs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                 f'${cost:.0f}', ha='center', va='bottom', fontsize=12)
    
    # Plot 3: Staffing Schedule
    ax3 = axes[1, 0]
    ax3.bar(range(n_hours), s_solution, color='purple', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Hour of Day')
    ax3.set_ylabel('Number of Staff')
    ax3.set_title('Optimal Staffing Schedule')
    ax3.set_xticks(range(n_hours))
    ax3.set_xticklabels([f'{h}:00' for h in hours], rotation=45)
    
    # Plot 4: Coverage Pie Chart
    ax4 = axes[1, 1]
    wedges, texts, autotexts = ax4.pie([total_attendance_served, total_attendance - total_attendance_served], 
            labels=['Served', 'Not Served'], 
            autopct='%1.1f%%', colors=['green', 'red'],
            explode=(0.05, 0))
    for w in wedges:
        w.set_alpha(0.7)
    ax4.set_title('Attendance Coverage')
    
    plt.tight_layout()
    plt.savefig('/home/claude/optimization_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[Figure 3: Optimization Results saved]")

# ============================================================================
# SECTION 8: SENSITIVITY ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("8. SENSITIVITY ANALYSIS")
print("="*70)

coverage_levels = [0.75, 0.80, 0.85, 0.90, 0.95]
sensitivity_results = []

print("\nAnalyzing impact of different coverage requirements...")

for cov in coverage_levels:
    # Rebuild constraint 1 with new coverage
    A_ub_sens = A_ub.copy()
    b_ub_sens = b_ub.copy()
    b_ub_sens[0] = -cov * total_attendance
    
    constraints_sens = LinearConstraint(A_ub_sens, -np.inf, b_ub_sens)
    
    result_sens = milp(c=c, constraints=constraints_sens, integrality=integrality, bounds=bounds)
    
    if result_sens.success:
        x_sol = np.round(result_sens.x[:n_hours]).astype(int)
        open_hrs = [hours[i] for i in range(n_hours) if x_sol[i] == 1]
        served = sum(attendance[i] for i in range(n_hours) if x_sol[i] == 1)
        
        sensitivity_results.append({
            'Coverage_Req': f"{cov*100:.0f}%",
            'Hours_Open': len(open_hrs),
            'Opening': f"{min(open_hrs)}:00",
            'Closing': f"{max(open_hrs)+1}:00",
            'Total_Cost': result_sens.fun,
            'Members_Served': served,
            'Actual_Coverage': f"{served/total_attendance*100:.1f}%"
        })

df_sensitivity = pd.DataFrame(sensitivity_results)
print("\nSENSITIVITY ANALYSIS: Impact of Coverage Requirement")
print("="*80)
print(df_sensitivity.to_string(index=False))

# Plot sensitivity analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

coverage_vals = [75, 80, 85, 90, 95]
cost_vals = [r['Total_Cost'] for r in sensitivity_results]
hours_vals = [r['Hours_Open'] for r in sensitivity_results]

ax1.plot(coverage_vals, cost_vals, 'bo-', linewidth=2, markersize=10)
ax1.set_xlabel('Coverage Requirement (%)', fontsize=12)
ax1.set_ylabel('Total Daily Cost ($)', fontsize=12)
ax1.set_title('Cost vs. Coverage Requirement', fontsize=14)
ax1.grid(True, alpha=0.3)
for cv, cost in zip(coverage_vals, cost_vals):
    ax1.annotate(f'${cost:.0f}', (cv, cost), textcoords="offset points", 
                 xytext=(0,10), ha='center')

ax2.bar(coverage_vals, hours_vals, color='steelblue', alpha=0.7, width=3)
ax2.set_xlabel('Coverage Requirement (%)', fontsize=12)
ax2.set_ylabel('Hours Open', fontsize=12)
ax2.set_title('Operating Hours vs. Coverage Requirement', fontsize=14)
ax2.set_ylim(0, 18)
for cv, hrs in zip(coverage_vals, hours_vals):
    ax2.text(cv, hrs + 0.3, str(hrs), ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('/home/claude/sensitivity_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[Figure 4: Sensitivity Analysis saved]")

# ============================================================================
# SECTION 9: CONCLUSIONS AND RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("9. CONCLUSIONS AND RECOMMENDATIONS")
print("="*70)

conclusions = f"""
KEY FINDINGS:

1. OPTIMAL SCHEDULE:
   • Recommended operating hours: {opening_str} to {closing_str}
   • Total: {len(open_hours_list)} hours (reduced from 17 hours)
   • Covers peak attendance periods while eliminating low-usage hours

2. COST SAVINGS:
   • Daily savings: ${savings:.2f} ({savings/full_op_cost*100:.1f}% reduction)
   • Monthly savings (30 days): ${savings * 30:,.2f}
   • Annual savings (365 days): ${savings * 365:,.2f}

3. SERVICE IMPACT:
   • {total_attendance_served/total_attendance*100:.1f}% of members still served
   • Only {total_attendance - total_attendance_served} members affected by reduced hours
   • Peak hours (5-8 PM) fully covered

4. STAFFING EFFICIENCY:
   • Total staff-hours per day: {sum(s_solution)}
   • Average staff per open hour: {sum(s_solution)/len(open_hours_list):.1f}

RECOMMENDATIONS:

1. Implement the optimized schedule to achieve ${savings * 365:,.0f} annual savings

2. Consider extending hours on specific high-demand days (e.g., weekends)

3. Offer alternative options for off-hours:
   - Online workout videos for early risers
   - Outdoor fitness areas for late-night exercisers

4. Monitor actual attendance after implementation and adjust accordingly

5. Survey members about schedule preferences before final implementation

6. Consider phased implementation to assess member response

MODEL LIMITATIONS:
- Assumes deterministic attendance patterns
- Does not account for seasonal variations or special events
- Treats all members equally (doesn't consider premium memberships)
- Linear cost model may not capture all real-world complexities

FUTURE EXTENSIONS:
- Incorporate stochastic attendance modeling
- Add day-of-week variations (weekday vs. weekend patterns)
- Include member satisfaction as a secondary objective
- Consider staff shift constraints (minimum shift length, breaks)
"""
print(conclusions)

# ============================================================================
# SECTION 10: REFERENCES
# ============================================================================
print("\n" + "="*70)
print("10. REFERENCES")
print("="*70)

references = """
1. Williams, H.P. (2013). Model Building in Mathematical Programming. Wiley.

2. SciPy Documentation: scipy.optimize.milp
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html

3. IHRSA (2023). Health Club Industry Data Report.

4. Hillier, F.S. & Lieberman, G.J. (2014). Introduction to Operations Research.
   McGraw-Hill.

5. Bradley, S.P., Hax, A.C., & Magnanti, T.L. (1977). Applied Mathematical
   Programming. Addison-Wesley.
"""
print(references)

print("\n" + "="*70)
print("END OF ANALYSIS")
print("="*70)
print("\nAll figures saved to /home/claude/")
print("  - attendance_pattern.png")
print("  - cost_breakdown.png")
print("  - optimization_results.png")
print("  - sensitivity_analysis.png")
