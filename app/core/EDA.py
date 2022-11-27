import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class EDA :
    
    @staticmethod
    def plotting_numeric_data (
    data:pd.DataFrame,
    numeric_columns:list,
    title_of_plot:str) :

        '''
            This method plot numeric data in two columns,
            when one contains a boxplot and other with histplot

            Parameters :
                data : A dataframe to plot numerica data
                numeric_columns : A previously filtered numeric data columns
                title_of_plot: A title to all plots    
        '''

        # Size of figure
        fig, ax = plt.subplots(figsize=(25,35))
        # Wall color
        color_wall = '#f5f5f5'
        fig.set_facecolor(color_wall)
        # Color palette
        #palette = sns.color_palette('flare', len(numeric_columns)*2)
        # Title
        plt.suptitle(
            title_of_plot, 
            fontsize=22,
            color='#404040',
            fontweight=600 )
        
        # Structure of plot
        lines = len(numeric_columns)
        number_of_columns_to_plot = 2 # Boxplot and Histplot
        position = 1 # Initial position of grid
        
        # Creating plot
        for column in numeric_columns :
            # Plot at the grid
            plt.subplot(lines, number_of_columns_to_plot, position)
            # Plot boxplot
            sns.boxplot(data=data, y=column)
            # Change position
            position+=1
            #plot at the grid
            plt.subplot(lines, number_of_columns_to_plot, position)
            # Plot histplot
            sns.histplot(data[column],kde=True, stat='density')
            # Change position
            position+=1
        # Adjust grid
        plt.subplots_adjust(top=0.95, hspace=0.3)
        return plt.show()
    
    @staticmethod
    def describe_column(
        dataframe:pd.DataFrame, column:str)->pd.Series:

        print(f'{column} statistics: \n\n{dataframe[column].describe()}')
    
    @staticmethod
    def verify_balance_data(
        dataframe:pd.DataFrame, 
        column:str, 
        dict_map_classes:dict,
        title_plot:str)->sns:

        processed_data = dataframe[column].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(12.5,6))

        for axis in ['top', 'right', 'left']:
            ax.spines[axis].set_color(None)
    
        for indice, value in enumerate(processed_data) :
            axis_x = dict_map_classes[indice]
            fig = plt.bar(axis_x,value,width=0.95,edgecolor='white',linewidth=0.95)
            ax.text(x=axis_x ,y=value, s=value ,horizontalalignment='center',
                    verticalalignment='bottom',fontdict={'fontsize':12.25})
        plt.title(title_plot, fontdict={'weight':'semibold', 'fontsize':20})

        return plt.show()
    
    @staticmethod
    def percentage_classes(
    dataframe:pd.DataFrame,
    column:str,
    dict_map_classes:dict)->str:

        for tuple in dataframe[column].map(
            dict_map_classes).value_counts().items() :

            print(f'Percentage of {tuple[0]}: {round(tuple[1]/len(dataframe)*100, 1)}%')