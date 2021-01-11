ebm = ExplainableBoostingClassifier(random_state=seed)
ebm.fit(X_train, y_train)