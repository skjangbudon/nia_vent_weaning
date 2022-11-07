# STEPWISE Feature Selection
import time, statsmodels
import matplotlib.pyplot as plt
import statsmodels.api as sm



def stepwise_fs(train_x, train_y, p_value):

    '''
        Input : Train data (train_x, train_y)
        Return : selected feature list 
    '''

    ''' time elapsed '''
    st_time = time.time()

    ''' Input Data '''
    features = list(train_x.keys())      # total features
    selected_total = []

    ''' Options '''
    sl_enter =  p_value        ## Set significance level (0.05 ~ 0.1)
    sl_remove =  p_value       ## Set significance level (0.05 ~ 0.1)

    variables = features

    # Feature Selection
    y = list(train_y) ## label
    sv_per_step = [] ## selected var in each step 
    adjusted_r_squared = [] ## 각 스텝별 수정된 결정계수
    steps = [] ## 스텝
    step = 0
    selected_variables = [] ## selected variable list

    while len(variables) > 0:
        remainder = list(set(variables) - set(selected_variables))
        pval = pd.Series(index=remainder) ## 변수의 p-value
        ## 기존에 포함된 변수와 새로운 변수 하나씩 돌아가면서 선형 모형 fitting
        for col in remainder: 
            X = train_x[selected_variables+[col]]
            X = sm.add_constant(X)
            model = sm.OLS(y,X).fit()
            pval[col] = model.pvalues[col]

        min_pval = pval.min()
        if min_pval < sl_enter: ## 최소 p-value 값이 기준 값보다 작으면 포함
            selected_variables.append(pval.idxmin())
            ## 선택된 변수들에대해서 어떤 변수를 제거할지 선택
            while len(selected_variables) > 0:
                selected_X = train_x[selected_variables]
                selected_X = sm.add_constant(selected_X)
                selected_pval = sm.OLS(y,selected_X).fit().pvalues[1:] ## 절편항의 p-value는 제거
                max_pval = selected_pval.max()
                if max_pval >= sl_remove: ## 최대 p-value값이 기준값보다 크거나 같으면 제외
                    remove_variable = selected_pval.idxmax()
                    selected_variables.remove(remove_variable)
                else:
                    break
            step += 1
            steps.append(step)
            adj_r_squared = sm.OLS(y,sm.add_constant(train_x[selected_variables])).fit().rsquared_adj
            adjusted_r_squared.append(adj_r_squared)
            sv_per_step.append(selected_variables.copy())
        else:
            break

    selected_total += selected_variables

    print('Completed...!')
    print('Elapsed time : {:.0f} sec'.format(time.time() - st_time))
    
    return selected_total