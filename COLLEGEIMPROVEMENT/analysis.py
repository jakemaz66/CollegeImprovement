from COLLEGEIMPROVEMENT.models import regression3

reg = regression3.reg

#Obtaning Weights of the Given Features
for feature, coefficient in zip(regression3.X.columns, reg.coef_):
    print(f"{feature}: {coefficient}")