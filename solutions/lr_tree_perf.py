lr_perf = ROC(lr.predict_proba).explain_perf(X_test_bin, y_test, name='Logistic Regression')
tree_perf = ROC(tree.predict_proba).explain_perf(X_test_bin, y_test, name='Classification Tree')