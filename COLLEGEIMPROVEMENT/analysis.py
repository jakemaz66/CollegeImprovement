from COLLEGEIMPROVEMENT.models import regression4

reg = regression4.reg

#Obtaning Weights of the Given Features
for feature, coefficient in zip(regression4.X.columns, reg.coef_):
    print(f"{feature}: {coefficient}")