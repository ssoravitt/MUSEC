from bokeh.plotting import figure, show
from bokeh.models import  ColumnDataSource, HoverTool, Div, CustomJS
from bokeh.layouts import gridplot, column, row, layout
from bokeh.models.widgets import  Dropdown
from bokeh.embed import components
from bokeh.resources import INLINE

from django.shortcuts import render
from django.contrib.auth.decorators import login_required


import numpy as np
import pandas as pd

@login_required(login_url='/login/')
def Sleep(request):

    def readCSV(csvname):
        Data = pd.read_csv(csvname)
        return Data

    Data = readCSV("Fitbit.csv")

    S1 = (Data[(Data['Subjects'] == 'S10')]).loc[:,['Timestamp',
                                                                    'Subjects',
                                                                    'HR_Baseline',
                                                                    'HR_Fitbit',
                                                                    'PredictionHR_Fitbit',
                                                                    'State']].sort_values('Timestamp')

    S2 = (Data[(Data['Subjects'] == 'S6')]).loc[:,['Timestamp',
                                                                    'Subjects',
                                                                    'HR_Baseline',
                                                                    'HR_Fitbit',
                                                                    'PredictionHR_Fitbit',
                                                                    'State']].sort_values('Timestamp')

    S3 = (Data[(Data['Subjects'] == 'S19')]).loc[:,['Timestamp',
                                                                    'Subjects',
                                                                    'HR_Baseline',
                                                                    'HR_Fitbit',
                                                                    'PredictionHR_Fitbit',
                                                                    'State']].sort_values('Timestamp')



    def stageIntensity(SubjectNo):
        SubjectIntensity = (SubjectNo[(SubjectNo['State'] == 'Intensity')]).loc[:,:]
        SubjectIntensity = SubjectIntensity.reset_index(drop=True)
        return SubjectIntensity
        
        
    def stageSleep(SubjectNo):
        SubjectSleep = (SubjectNo[(SubjectNo['State'] == 'Sleep')]).loc[:,:]
        SubjectSleep = SubjectSleep.reset_index(drop=True)
        
        
        return SubjectSleep

    def stageResting(SubjectNo):
        SubjectResting = (SubjectNo[(SubjectNo['State'] == 'Resting')]).loc[:,:]
        SubjectResting = SubjectResting.reset_index(drop=True)

        
        return SubjectResting

    Sleep_S1_df = stageSleep(S1)
    Sleep_S2_df = stageSleep(S2)
    Sleep_S3_df = stageSleep(S3)

    Resting_S1_df = stageResting(S1)
    Resting_S2_df = stageResting(S2)
    Resting_S3_df = stageResting(S3)

    Intensity_S1_df = stageIntensity(S1)
    Intensity_S2_df = stageIntensity(S2)
    Intensity_S3_df = stageIntensity(S3)

    class BlandAltman():

        def __init__(self, data1, data2):
            self.data1 = data1
            self.data2 = data2

        def mean(self):
            BlAlt_mean = np.mean([self.data1, self.data2], axis = 0)
            return BlAlt_mean

        def diff(self):
            BlAlt_diff = np.array(self.data1 - self.data2) # y-axis
            return BlAlt_diff

        def md(self):
            BlAlt_md = np.mean(self.diff()) # Mean Line
            return BlAlt_md

        def sd(self):
            BlAlt_sd = np.std(self.diff(), axis = 0)
            return BlAlt_sd

        def sdUpper(self):
            sd_Upper = self.md() + (self.sd() * 1.96) # SD Upper line
            return sd_Upper

        def sdLower(self):
            sd_Lower = self.md() - (self.sd() * 1.96) # SD Upper line
            return sd_Lower

    def PushBland(SubjectNo, Baseline, Device, PredictDevice):
        BlandValue = BlandAltman(SubjectNo[Baseline], SubjectNo[Device])
        SubjectNo['Mean'] = BlandValue.mean()
        SubjectNo['Diff'] = BlandValue.diff()

        BlandValuePredict = BlandAltman(SubjectNo[Baseline], SubjectNo[PredictDevice])
        SubjectNo['Mean_Predict'] = BlandValuePredict.mean()
        SubjectNo['Diff_Predict'] = BlandValuePredict.diff()

        SubjectNo['SD_Upper'] = BlandValue.sdUpper()
        SubjectNo['SD_Lower'] = BlandValue.sdLower()
        SubjectNo['MD'] = BlandValue.md()
        SubjectNo['SD_UpperPredict'] = BlandValuePredict.sdUpper()
        SubjectNo['SD_LowerPredict'] = BlandValuePredict.sdLower()
        SubjectNo['MD_Predict'] = BlandValuePredict.md()

        x = range(0,500)

        S_Cds = ColumnDataSource(SubjectNo)
        S_Cds.add(x, 'X_SD')

        return S_Cds


    Sleep_S1 = PushBland(Sleep_S1_df, 'HR_Baseline', 'HR_Fitbit', 'PredictionHR_Fitbit')
    Sleep_S2 = PushBland(Sleep_S2_df, 'HR_Baseline', 'HR_Fitbit', 'PredictionHR_Fitbit')
    Sleep_S3 = PushBland(Sleep_S3_df, 'HR_Baseline', 'HR_Fitbit', 'PredictionHR_Fitbit')

    Resting_S1 = PushBland(Resting_S1_df, 'HR_Baseline', 'HR_Fitbit', 'PredictionHR_Fitbit')
    Resting_S2 = PushBland(Resting_S2_df, 'HR_Baseline', 'HR_Fitbit', 'PredictionHR_Fitbit')
    Resting_S3 = PushBland(Resting_S3_df, 'HR_Baseline', 'HR_Fitbit', 'PredictionHR_Fitbit')

    Intensity_S1 = PushBland(Intensity_S1_df, 'HR_Baseline', 'HR_Fitbit', 'PredictionHR_Fitbit')
    Intensity_S2 = PushBland(Intensity_S2_df, 'HR_Baseline', 'HR_Fitbit', 'PredictionHR_Fitbit')
    Intensity_S3 = PushBland(Intensity_S3_df, 'HR_Baseline', 'HR_Fitbit', 'PredictionHR_Fitbit')

    fill_source = PushBland(Sleep_S1_df, 'HR_Baseline', 'HR_Fitbit', 'PredictionHR_Fitbit')

    toolList = ['tap', 'box_zoom', 'reset', 'save']

    tooltips = [

        ('Times' , '@index'),
        ('HR_Baseline', '@HR_Baseline'),
        ('HR Fitbit', '@HR_Fitbit'),
        ('HR Fitbit Prediction', '@PredictionHR_Fitbit')
    ]

    tooltips_Bland = [
        ('Times' , '@index'),
        ('HR_Baseline', '@HR_Baseline'),
        ('HR Fitbit', '@HR_Fitbit'),
        ('HR Fitbit Prediction', '@PredictionHR_Fitbit'),
        ('Mean', '@Mean'),
        ('Diff', '@Diff')
    ]

    tooltips_BlandPredict = [
        ('Times' , '@index'),
        ('HR_Baseline', '@HR_Baseline'),
        ('HR Fitbit', '@HR_Fitbit'),
        ('HR Fitbit Prediction', '@PredictionHR_Fitbit'),
        ('Mean Prediction', '@Mean_Predict'),
        ('Diff Prediction', '@Diff_Predict')
    ]

    toolList_Scatter = ['lasso_select', 'box_zoom', 'tap', 'reset', 'save']

    Fig = figure(plot_height = 400, plot_width = 1110,
                    title = 'Heart rate from devices ', tools = toolList, toolbar_location = 'right',
                    x_axis_label = 'Times', y_axis_label = 'Heart rate')

    PolarLine = Fig.line( x = 'index', y = 'HR_Baseline', line_width = 3, source = fill_source, color='#CE1141', legend='Baseline')
    Fig.line(x = 'index', y = 'HR_Fitbit', line_width = 3, source = fill_source, color='#007A33', legend='Fitbit')
    Fig.line(x = 'index', y ='PredictionHR_Fitbit', line_width = 3, source = fill_source, color='#006BB6', legend='Fitbit Correction')


    Fig.background_fill_color = "#fafafa"
    Fig.legend.click_policy = 'hide'

    Fig.add_tools(HoverTool(tooltips = tooltips, renderers = [PolarLine], mode = 'vline'))

    menu = [("Mr.A : Sleep", "Sleep_S1"), ("Mr.B : Sleep", "Sleep_S2"), ("Mr.C : Sleep", "Sleep_S3"), None,
            ("Mr.A : Resting", "Resting_S1"),("Mr.B : Resting", "Resting_S2"),("Mr.C : Resting", "Resting_S3"), None,
            ("Mr.A : Intensity", "Intensity_S1"),("Mr.B : Intensity", "Intensity_S2"),("Mr.C : Intensity", "Intensity_S3")]

    

    selectSubject = Dropdown(label = "Activity : ", button_type="warning", menu = menu, width=300, height=30)
    

    def Scatter():
        Scatter = figure(title = 'Bland Altman : Fitbit',
                            tools = toolList_Scatter,
                            plot_height = 350,
                            plot_width = 555,
                            x_axis_label='Mean',
                            y_axis_label='Diff',
                            y_range =(-100, 100))

        Scatter.circle(x = 'Mean', y = 'Diff', source = fill_source, size=8, fill_alpha = 0.4,line_width = 2, color = '#007A33')
        Scatter.line(x = 'X_SD', y = 'SD_Upper', color = "red", line_width = 5, alpha = 0.2, source = fill_source)
        Scatter.line(x = 'X_SD', y = 'SD_Lower', color = "red", line_width = 5, alpha = 0.2, source = fill_source)
        Scatter.line(x = 'X_SD', y = 'MD', color = "blue", line_width = 5, alpha = 0.2, source = fill_source)
        Scatter.background_fill_color = "#fafafa"
        Scatter.add_tools(HoverTool(tooltips = tooltips_Bland))

        return Scatter


    def PredictScatter():
        PredictScatter = figure(title = 'Bland Altman : Fitbit Correction',
                                    tools = toolList_Scatter,
                                    plot_height = 350,
                                    plot_width = 555,
                                    toolbar_location = 'right',
                                    x_axis_label='Mean',
                                    y_range =(-100, 100))

        PredictScatter.circle(x = 'Mean_Predict', y = 'Diff_Predict', source = fill_source, size = 8, fill_alpha = 0.4, line_width = 2, color = '#006BB6')
        PredictScatter.line(x = 'X_SD', y = 'SD_UpperPredict', color = "red", line_width = 5, alpha = 0.2, source = fill_source)
        PredictScatter.line(x = 'X_SD', y = 'SD_LowerPredict', color = "red", line_width = 5, alpha = 0.2, source = fill_source)
        PredictScatter.line(x = 'X_SD', y = 'MD_Predict', color = "blue", line_width = 5, alpha = 0.2, source = fill_source)
        PredictScatter.background_fill_color = "#fafafa"
        PredictScatter.add_tools(HoverTool(tooltips = tooltips_BlandPredict))

        return PredictScatter


    codeSubject = """
        var f = cb_obj.value;
        var sdata = source.data;
        var sleep1 = Sleep_S1.data;
        var sleep2 = Sleep_S2.data;
        var sleep3 = Sleep_S3.data;
        
        var resting1 = Resting_S1.data;
        var resting2 = Resting_S2.data;
        var resting3 = Resting_S3.data;
        
        var intensity1 = Intensity_S1.data;
        var intensity2 = Intensity_S2.data;
        var intensity3 = Intensity_S3.data;
        

        if (f == 'Sleep_S1') {
        for (key in sleep1)
        {
            sdata[key] = [];
            for (i = 0; i < sleep1[key].length; i++){
            sdata[key].push(sleep1[key][i])

            }
        }
        } 
        
        else if (f == 'Sleep_S2') {
        for (key in sleep2) {
            sdata[key] = [];
            for (i = 0; i < sleep2[key].length; i++){
            sdata[key].push(sleep2[key][i]);
            }
        }
        } 
        
        else if (f == 'Sleep_S3') {
        for (key in sleep3) {
            sdata[key] = [];
            for (i = 0; i < sleep3[key].length; i++){
            sdata[key].push(sleep3[key][i]);
            }
        }
        }  
        
        else if (f == 'Resting_S1') {
        for (key in resting1) {
            sdata[key] = [];
            for (i = 0; i < resting1[key].length; i++){
            sdata[key].push(resting1[key][i]);
            }
        }
        }  
        
        else if (f == 'Resting_S2') {
        for (key in resting2) {
            sdata[key] = [];
            for (i = 0; i < resting2[key].length; i++){
            sdata[key].push(resting2[key][i]);
            }
        }
        } 
        
        else if (f == 'Resting_S3') {
        for (key in resting3) {
            sdata[key] = [];
            for (i = 0; i < resting3[key].length; i++){
            sdata[key].push(resting3[key][i]);
            }
        }
        } 
        
        else if (f == 'Intensity_S1') {
        for (key in intensity1) {
            sdata[key] = [];
            for (i = 0; i < intensity1[key].length; i++){
            sdata[key].push(intensity1[key][i]);
            }
        }
        }  
        
        else if (f == 'Intensity_S2') {
        for (key in intensity2) {
            sdata[key] = [];
            for (i = 0; i < intensity2[key].length; i++){
            sdata[key].push(intensity2[key][i]);
            }
        }
        }
        
        else if (f == 'Intensity_S3') {
        for (key in intensity3) {
            sdata[key] = [];
            for (i = 0; i < intensity3[key].length; i++){
            sdata[key].push(intensity3[key][i]);
            }
        }
        }
        
        else {
        for (key in sleep1) {
            sdata[key] = [];
            for (i=0; i<sleep1[key].length; i++){
            sdata[key].push(sleep1[key][i]);
            }
        }

        }
        ;

        source.change.emit();
    """
    dict_source = {'source': fill_source,
                'Sleep_S1': Sleep_S1,
                'Sleep_S2': Sleep_S2,
                'Sleep_S3': Sleep_S3,
                'Resting_S1': Resting_S1,
                'Resting_S2': Resting_S2,
                'Resting_S3': Resting_S3,
                'Intensity_S1': Intensity_S1,
                'Intensity_S2': Intensity_S2,
                'Intensity_S3': Intensity_S3,}

    selectSubject.callback = CustomJS(args=dict_source, code=codeSubject)
    
    grid2 = gridplot([[Fig]])
    grid3 = gridplot([[Scatter(), PredictScatter()]])

    layout = (column(selectSubject, grid2 ,grid3))

    script, div = components(layout)
    
    return render(request, 'visra/sleep.html', {'resources' : INLINE.render(), 'script' : script, 'div' : div})


