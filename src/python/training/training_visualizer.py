import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


class TrainingVisualizer:
    def __init__(self, config: dict, training_results_dict: dict) -> None:
        self._config = config
        self._evaluate_metric = self._config["training"]["evaluate_metric"]
        self._training_results_dict = training_results_dict

    def plot_training_results(self, timestamp: int, save_plots: bool = False) -> bool:
        """
        :param timestamp: timestamp of the training-run
        :param save_plots: whether to save the plots in artifacts or not
        :return: boolean indicating that plotting was successful

        - makes plots to compare model results
        """

        # color dictionary for plots and legend:
        colors = (
            ["#3D5567", "#ED2C2A"]
            + ["gcmykbr"[i] for i in range(len(self._training_results_dict["algorithm_names"]) - 3)]
            + ["#A68B48"]
        )
        model_color_dict = dict(zip(self._training_results_dict["algorithm_names"], colors))

        # construct figure:
        fig, axs = plt.subplots(
            3,
            4,
            figsize=(25, 15),
            gridspec_kw={"width_ratios": (0.25, 0.25, 0.25, 0.25)},
        )
        fig.suptitle(
            "Training-plots",
            fontsize=16,
        )

        # make prediction plots:
        self._plot_predictions(axs[0, 0], model_color_dict)
        self._plot_test_predictions(axs[0, 2], model_color_dict)

        # make goodness of fit plot:
        self._plot_goodness_of_fit(axs[0, 1], model_color_dict)
        self._plot_test_goodness_of_fit(axs[0, 3], model_color_dict)

        # make residual plot:
        self._plot_residuals(axs[1, 0], model_color_dict)
        self._plot_test_residuals(axs[1, 2], model_color_dict)

        # boxplots with evaluation metric for cv:
        self._plot_boxplots(axs[1, 1], self._training_results_dict["algorithm_names"])
        self._plot_test_boxplots(axs[1, 3], self._training_results_dict["algorithm_names"])

        # cooks distance plot:
        self._plot_cooks_distance(axs[2, 0])

        # make feature importance plot:
        # self._plot_feature_importance(axs[2, 1], model_color_dict)

        # correlation plot between features and rbl
        # self._plot_correlation(axs[2, 2])

        # legend:
        patches = [mpatches.Patch(color=model_color_dict[model], label=model) for model in model_color_dict]
        patches = [mpatches.Patch(color="black", label="Training Data")] + patches
        fig.legend(handles=patches, loc="lower right")

        # set axes off:
        axs[2, 3].axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # plot or save the plots:
        if save_plots:
            file_name = f"training_plots_{timestamp}"
            os.makedirs("artifacts/plots/training", exist_ok=True)
            plt.savefig(f"artifacts/plots/training/{file_name}", bbox_inches=None, dpi=None)
            plt.close()
        else:
            plt.show()

        return True

    def _plot_predictions(self, ax: object, model_color_dict: dict) -> None:
        """
        :param ax: axes for plot
        :param model_color_dict: dictionary with the models and their colors

        - plots training-data-rbl and the predictions made by the models
        """
        ax.scatter(
            self._training_results_dict["data"]["x_train"].index,
            self._training_results_dict["data"]["y_train"],
            s=8,
            alpha=1,
            color="black",
        )
        for model_name in self._training_results_dict["algorithm_names"]:
            ax.scatter(
                self._training_results_dict["data"]["x_train"].index,
                self._training_results_dict[model_name]["y_pred"],
                s=self._config["training"]["plotting_params"]["s"],
                alpha=self._config["training"]["plotting_params"]["alpha"],
                color=model_color_dict[model_name],
            )
        ax.set(ylabel="Prediction Values Training Set", xlabel="Index")

    def _plot_test_predictions(self, ax: object, model_color_dict: dict) -> None:
        """
        :param ax: axes for plot
        :param model_color_dict: dictionary with the models and their colors

        - plots training-data-rbl and the predictions made by the models
        """
        ax.scatter(
            self._training_results_dict["data"]["x_test"].index,
            self._training_results_dict["data"]["y_test"],
            s=8,
            alpha=1,
            color="black",
        )
        for model_name in self._training_results_dict["algorithm_names"]:
            ax.scatter(
                self._training_results_dict["data"]["x_test"].index,
                self._training_results_dict[model_name]["y_test_pred"],
                s=self._config["training"]["plotting_params"]["s"],
                alpha=self._config["training"]["plotting_params"]["alpha"],
                color=model_color_dict[model_name],
            )
        ax.set(ylabel="Prediction Values Test Set", xlabel="Index")

    def _plot_goodness_of_fit(self, ax: object, model_color_dict: dict) -> None:
        """
        :param ax: axes for plot
        :param model_color_dict: dictionary with the models and their colors

        - plots the predicted values against the true values
        """
        for model_name in self._training_results_dict["algorithm_names"]:
            ax.scatter(
                self._training_results_dict["data"]["y_train"],
                self._training_results_dict[model_name]["y_pred"],
                s=self._config["training"]["plotting_params"]["s"],
                alpha=self._config["training"]["plotting_params"]["alpha"],
                color=model_color_dict[model_name],
            )
        self._plot_diagonal_line(ax)
        ax.set(ylabel="Prediction Values Training Set", xlabel="True Values")

    def _plot_test_goodness_of_fit(self, ax: object, model_color_dict: dict) -> None:
        """
        :param ax: axes for plot
        :param model_color_dict: dictionary with the models and their colors

        - plots the predicted values against the true values
        """
        for model_name in self._training_results_dict["algorithm_names"]:
            ax.scatter(
                self._training_results_dict["data"]["y_test"],
                self._training_results_dict[model_name]["y_test_pred"],
                s=self._config["training"]["plotting_params"]["s"],
                alpha=self._config["training"]["plotting_params"]["alpha"],
                color=model_color_dict[model_name],
            )
        self._plot_diagonal_line(ax)
        ax.set(ylabel="Prediction Values Test Set", xlabel="True Values")

    def _plot_residuals(self, ax: object, model_color_dict: dict) -> None:
        """
        :param ax: axes for plot
        :param model_color_dict: dictionary with the models and their colors

        - plots the residuals of the prediction
        """
        for model_name in self._training_results_dict["algorithm_names"]:
            ax.scatter(
                self._training_results_dict["data"]["x_train"].index,
                self._training_results_dict[model_name]["y_train_residuals"],
                s=self._config["training"]["plotting_params"]["s"],
                alpha=self._config["training"]["plotting_params"]["alpha"],
                color=model_color_dict[model_name],
            )
        ax.axhline(0, color="lightgray", linestyle="--")
        ax.set(ylabel="Residuals Training Set", xlabel="Index")

    def _plot_test_residuals(self, ax: object, model_color_dict: dict) -> None:
        """
        :param ax: axes for plot
        :param model_color_dict: dictionary with the models and their colors

        - plots the residuals of the prediction
        """
        for model_name in self._training_results_dict["algorithm_names"]:
            ax.scatter(
                self._training_results_dict["data"]["x_test"].index,
                self._training_results_dict[model_name]["y_test_residuals"],
                s=self._config["training"]["plotting_params"]["s"],
                alpha=self._config["training"]["plotting_params"]["alpha"],
                color=model_color_dict[model_name],
            )
        ax.axhline(0, color="lightgray", linestyle="--")
        ax.set(ylabel="Residuals Test Set", xlabel="Index")

    def _plot_boxplots(self, ax: object, names: list) -> None:
        """
        :param ax: axes for plot
        :param names: names of the algorithms

        - plots boxplots of the cross-validation-performance of the algorithms
        """
        ax.boxplot(self._training_results_dict["cv_results"])

        ax.axhline(0, color="lightgray", linestyle="--")
        ax.set(
            xticklabels=names,
            ylabel=self._evaluate_metric.upper(),
            xlabel="Algorithms Training Set",
        )
        ax.tick_params("x", labelsize=12, labelrotation=20)

    def _plot_test_boxplots(self, ax: object, names: list) -> None:
        """
        :param ax: axes for plot
        :param names: names of the algorithms

        - plots boxplots of the cross-validation-performance of the algorithms
        """
        ax.boxplot(self._training_results_dict["test_results"])

        ax.axhline(0, color="lightgray", linestyle="--")
        ax.set(
            xticklabels=names,
            ylabel=self._evaluate_metric.upper(),
            xlabel="Algorithms Test Set",
        )
        ax.tick_params("x", labelsize=12, labelrotation=20)

    def _plot_cooks_distance(self, ax: object) -> None:
        """
        :param ax: axes for plot

        - plots the cooks-distance, an indicator of how influencial each sample is
        """
        (markers, stem_lines, _) = ax.stem(
            self._training_results_dict["cooks_distance_data"].index,
            self._training_results_dict["cooks_distance"][0],
            markerfmt=" ",
            basefmt=" ",
        )
        plt.setp(markers, marker="o", color="black", markersize=2)
        plt.setp(stem_lines, linestyle="-", color="black", linewidth=1)

        ax.axhline(0, color="lightgray", linestyle="--")

        ax.set(xlabel="Index", ylabel="Cooks Distance")

    def _plot_feature_importance(self, ax: object, model_color_dict: dict) -> None:
        """
        :param ax: axes for plot
        :param model_color_dict: dictionary with the models and their colors

        - plots the feature importance for the models
        """
        for model_name in self._training_results_dict["algorithm_names"]:
            ax.scatter(
                self._training_results_dict[model_name]["feature_rank_df"]["rank"],
                self._training_results_dict[model_name]["feature_rank_df"]["column_name"],
                color=model_color_dict[model_name],
            )

        ax.set(ylabel="Column Name", xlabel="Rank (Ensemble as Reference)")
        ax.tick_params("y_train", labelsize=8)

    def _plot_correlation(self, ax: object) -> None:
        """
        :param ax: axes for plot

        - plots the correlation of the features to the rbl
        - orders them by the feature-order of the plot_feature_importance-method
        """
        if self._training_results_dict["rbl_corr"].isnull().all().all():
            ax.text(0, 0.5, "No correlation available")
        else:
            ax.scatter(
                self._training_results_dict["rbl_corr"],
                self._training_results_dict["rbl_corr"].index,
                c=self._training_results_dict["rbl_corr"],
                cmap="cool",
            )
        ax.axvline(0, color="lightgray")

        ax.set(ylabel="Column Name", xlabel="Correlation to Target", xlim=(-1, 1))
        ax.tick_params("y_train", labelsize=8)

    def _plot_diagonal_line(self, ax: object) -> None:
        """
        :param ax: axes for plot

        # plots diagonal line
        """
        min_axis_value = min(ax.get_ylim()[0], ax.get_ylim()[0])
        max_axis_value = max(ax.get_ylim()[1], ax.get_ylim()[1])
        ax.plot(
            [min_axis_value, max_axis_value],
            [min_axis_value, max_axis_value],
            color="lightgrey",
            linestyle="--",
        )
