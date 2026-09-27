import subprocess
import sys


# ============================================================
# HEADER
# ============================================================

def header():
    print("\n" + "=" * 60)
    print("       AI FINANCIAL INTELLIGENCE")
    print("=" * 60)


# ============================================================
# RUN PROGRAM
# ============================================================

def run_program(file):

    subprocess.run(
        [sys.executable, file]
    )


# ============================================================
# MAIN MENU
# ============================================================

while True:

    header()

    print("\n1. Stock Price Prediction")
    print("2. Company Growth Prediction")
    print("3. Risk Analysis")
    print("4. Investment Recommendation")
    print("5. Exit")

    choice = input("\nEnter your choice: ").strip()


    # --------------------------------------------------------
    # STOCK PREDICTION
    # --------------------------------------------------------

    if choice == "1":

        run_program(
            "src/prediction/predict_stock.py"
        )


    # --------------------------------------------------------
    # COMPANY GROWTH
    # --------------------------------------------------------

    elif choice == "2":

        run_program(
            "src/prediction/predict_company_growth.py"
        )


    # --------------------------------------------------------
    # RISK ANALYSIS
    # --------------------------------------------------------

    elif choice == "3":

        run_program(
            "src/risk/risk_analysis.py"
        )


    # --------------------------------------------------------
    # INVESTMENT RECOMMENDATION
    # --------------------------------------------------------

    elif choice == "4":

        run_program(
            "src/recommendation/investment_recommendation.py"
        )


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    elif choice == "5":

        print("\nThank you for using")
        print("AI Financial Intelligence.")

        break


    else:

        print("\nInvalid choice.")

    
    input("\nPress ENTER to return to the main menu...")