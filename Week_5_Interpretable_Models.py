import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Week 5 - Interpretable (Whitebox) Models
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns

    return (
        DecisionTreeClassifier,
        LinearRegression,
        LogisticRegression,
        accuracy_score,
        np,
        plot_tree,
        sns,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Step 1: Load and Prepare the Data
    """)
    return


@app.cell
def _(sns):
    # Load Titanic dataset
    titanic = sns.load_dataset('titanic')

    # Select and clean features
    titanic = titanic[['survived', 'age', 'sex', 'pclass']].dropna()
    titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})

    titanic.head(10)
    return (titanic,)


@app.cell
def _(titanic, train_test_split):
    X = titanic[['age', 'sex', 'pclass']]
    y = titanic['survived']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X, X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Splitting nodes based on Gini coefficient
    ### 1. Gini Impurity
    The Gini impurity is defined as:

    \[
    \text{Gini} = 1 - \sum_{i=1}^{c} p_i^2
    \]

    Where \( p_i \) is the proportion of class \( i \) in the node.

    ### 2. Weighted Gini Impurity (After Split)
    When a node is split into a Left child and a Right child, the quality of the split is measured by the weighted average of the children's impurity:

    \[
    \text{Gini}_{\text{weighted}} = \frac{N_{\text{left}}}{N_{\text{total}}} \cdot \text{Gini}_{\text{left}} + \frac{N_{\text{right}}}{N_{\text{total}}} \cdot \text{Gini}_{\text{right}}
    \]

    Where:
    *   \( N_{\text{left}} \) = number of samples in the left node.
    *   \( N_{\text{right}} \) = number of samples in the right node.
    *   \( N_{\text{total}} \) = total number of samples in the parent node.

    ### 3. Information Gain
    The reduction in impurity achieved by the split. The algorithm chooses the split with the highest gain:

    \[
    \text{Gain} = \text{Gini}_{\text{parent}} - \text{Gini}_{\text{weighted}}
    \]



    ---

    ## Calculation for 10 first instances (passengers) in the Titanic dataset

    ### Task 1: Calculate Gini Impurity of the Root Node


    ---

    ### Task 2: Evaluate Split on Feature "Sex" by calculating the information gain for the split on sex


    ---

    ### Task 3: Evaluate Split on Feature "Age" (Threshold ≤ 30)


    ---

    ### Task 4: Comparison and Decision: which split will the decision tree algorithm use?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: Verify Manual Calculation in Code
    """)
    return


@app.cell
def _(np, titanic):
    def gini_impurity(y):
        """Calculate Gini impurity for a binary classification node."""
        if len(y) == 0:
            return 0
        p1 = np.sum(y == 1) / len(y)
        p0 = 1 - p1
        return 1 - (p0**2 + p1**2)

    def information_gain(y, y_left, y_right):
        """Calculate information gain from a split."""
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)

        if n_left == 0 or n_right == 0:
            return 0

        parent_gini = gini_impurity(y)
        child_gini = (n_left / n) * gini_impurity(y_left) + (n_right / n) * gini_impurity(y_right)

        return parent_gini - child_gini

    def find_best_split(X, y):
        """Find the best feature and threshold for splitting."""
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in X.columns:
            thresholds = np.unique(X[feature])
            for threshold in thresholds:
                left_mask = X[feature] <= threshold
                right_mask = ~left_mask

                gain = information_gain(y, y[left_mask], y[right_mask])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    # Test on first 10 rows (matching manual calculation)
    first_10 = titanic.head(10)
    X_10 = first_10[['age', 'sex', 'pclass']]
    y_10 = first_10['survived']

    feature, threshold, gain = find_best_split(X_10, y_10)
    print(f"Best split on first 10 rows: {feature} <= {threshold:.2f}")
    print(f"Information Gain: {gain:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Step 3: Train Decision Tree and Linear Regression Models
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    LinearRegression,
    LogisticRegression,
    X_test,
    X_train,
    accuracy_score,
    y_test,
    y_train,
):
    # 1. Decision Tree Classifier
    dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, y_pred_dt)

    # 2. Linear Regression (Used as a classifier by thresholding at 0.5)
    # Note: Linear Regression is not ideal for classification but used here for comparison
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, y_train)
    y_pred_lin = lin_reg_model.predict(X_test)
    y_pred_lin_class = (y_pred_lin >= 0.5).astype(int)
    lin_accuracy = accuracy_score(y_test, y_pred_lin_class)

    # 3. Logistic Regression (Proper linear baseline for classification)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, y_pred_lr)

    print(f"\n=== Model Comparison ===")
    print(f"Decision Tree Accuracy:       {dt_accuracy:.4f}")
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    print(f"Linear Regression Accuracy:   {lin_accuracy:.4f}")
    return dt_accuracy, dt_model, lin_accuracy, lr_accuracy, lr_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Step 4: Model Comparison and Visualization
    """)
    return


@app.cell
def _(
    X,
    dt_accuracy,
    dt_model,
    lin_accuracy,
    lr_accuracy,
    lr_model,
    plot_tree,
):
    def _():

        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.inspection import PartialDependenceDisplay

        # Set modern styling
        sns.set_style("white")
        sns.set_context("notebook", font_scale=1.1)

        # Accessible color palette (colorblind-friendly)
        palette = {
            'Decision Tree': '#2E86AB',       # Blue
            'Logistic Regression': '#A23B72', # Purple
            'Linear Regression': '#F18F01',   # Orange
            'Feature': '#44AF69'              # Green
        }

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

        # =============================================================================
        # Plot 1: Combined Feature Importances (Decision Tree & Logistic Regression)
        # =============================================================================
        ax1 = fig.add_subplot(gs[0, 0])

        # Prepare data for comparison
        features = X.columns
        dt_importances = dt_model.feature_importances_
        lr_coefficients = np.abs(lr_model.coef_[0])  # Use absolute values for comparison

        # Normalize both to 0-1 scale for fair comparison
        dt_norm = dt_importances / dt_importances.max()
        lr_norm = lr_coefficients / lr_coefficients.max()

        # Create grouped bar chart
        x = np.arange(len(features))
        width = 0.35

        bars1 = ax1.bar(x - width/2, dt_norm, width, label='Decision Tree', 
                        color=palette['Decision Tree'], edgecolor='white', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, lr_norm, width, label='Logistic Regression', 
                        color=palette['Logistic Regression'], edgecolor='white', linewidth=1.5)

        ax1.set_xticks(x)
        ax1.set_xticklabels(features, fontsize=11)
        ax1.set_ylabel('Normalized Importance', fontsize=11, fontweight='bold')
        ax1.set_title('Feature Importance Comparison', fontsize=13, fontweight='bold', pad=10)
        ax1.legend(frameon=False, loc='upper right')
        ax1.set_ylim(0, 1.1)
        ax1.axhline(0, color='black', linewidth=0.5, alpha=0.3)

        # Remove spines
        for spine in ['top', 'right', 'left']:
            ax1.spines[spine].set_visible(False)
        ax1.spines['bottom'].set_color('#DDDDDD')

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=9)

        # =============================================================================
        # Plot 2: Model Accuracy Comparison
        # =============================================================================
        ax2 = fig.add_subplot(gs[0, 1])

        models = ['Decision\nTree', 'Logistic\nRegression', 'Linear\nRegression']
        accuracies = [dt_accuracy, lr_accuracy, lin_accuracy]
        colors = [palette['Decision Tree'], palette['Logistic Regression'], palette['Linear Regression']]

        bars = ax2.bar(models, accuracies, color=colors, edgecolor='white', linewidth=1.5, width=0.6)

        ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax2.set_title('Model Performance Comparison', fontsize=13, fontweight='bold', pad=10)
        ax2.set_ylim(0, 1.0)

        # Remove spines
        for spine in ['top', 'right', 'left']:
            ax2.spines[spine].set_visible(False)
        ax2.spines['bottom'].set_color('#DDDDDD')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')


        # =============================================================================
        # Plot 3: Decision Tree Structure
        # =============================================================================
        ax5 = fig.add_subplot(gs[1, :])

        plot_tree(dt_model, feature_names=X.columns, 
                  class_names=['Died', 'Survived'], 
                  filled=True, ax=ax5, fontsize=9, 
                  rounded=True, precision=2,
                  impurity=False)

        ax5.set_title('Decision Tree Structure (max_depth=3)', fontsize=13, fontweight='bold', pad=10)

        plt.show()

    _()
    return


if __name__ == "__main__":
    app.run()
