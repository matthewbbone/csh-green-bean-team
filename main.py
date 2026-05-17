import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont


NETWORK_FILE_OPTIONS = {
    "geo": ("geo_net.csv", "uk_geo_net.csv"),
    "industry": ("ind_net.csv", "uk_ind_net.csv"),
    "skill": ("skill_net.csv", "uk_skill_net.csv"),
    "occupation": ("occ_net.csv", "uk_occ_net.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regress occupation transitions on geographic, industry, and skill similarity networks."
    )
    parser.add_argument(
        "--networks-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "networks",
        help="Directory containing the geographic, industry, skill, and occupation transition network CSVs.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Directory where r2_bar_chart.html and r2_bar_chart.png will be saved.",
    )
    return parser.parse_args()


def load_network(path: Path) -> pd.DataFrame:
    network = pd.read_csv(path, index_col=0)
    network.index = network.index.astype(str)
    network.columns = network.columns.astype(str)
    return network.apply(pd.to_numeric)


def resolve_network_file(networks_dir: Path, file_options: tuple[str, ...]) -> Path:
    for file_name in file_options:
        path = networks_dir / file_name
        if path.exists():
            return path
    expected = ", ".join(file_options)
    raise FileNotFoundError(f"Expected one of these files in {networks_dir}: {expected}")


def align_networks(networks: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    labels = networks["occupation"].index.intersection(networks["occupation"].columns)
    for name, network in networks.items():
        labels = labels.intersection(network.index).intersection(network.columns)
        if network.shape[0] != network.shape[1]:
            raise ValueError(f"{name} network is not square: {network.shape}")

    if labels.empty:
        raise ValueError("No common occupation IDs found across all network matrices.")

    labels = labels.sort_values()
    return {name: network.loc[labels, labels] for name, network in networks.items()}


def flatten_networks(networks: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
    occ = networks["occupation"]
    mask = np.ones(occ.shape, dtype=bool)
    np.fill_diagonal(mask, False)

    predictors = pd.DataFrame(
        {
            "geo": networks["geo"].to_numpy()[mask],
            "industry": networks["industry"].to_numpy()[mask],
            "skill": networks["skill"].to_numpy()[mask],
        }
    )
    target = pd.Series(occ.to_numpy()[mask], name="occupation_transitions")

    valid = predictors.notna().all(axis=1) & target.notna()
    return predictors.loc[valid].reset_index(drop=True), target.loc[valid].reset_index(drop=True)


def fit_r_squared(x: pd.DataFrame, y: pd.Series) -> tuple[float, np.ndarray]:
    design = np.column_stack([np.ones(len(x)), x.to_numpy()])
    beta, *_ = np.linalg.lstsq(design, y.to_numpy(), rcond=None)
    y_hat = design @ beta

    residual_sum_squares = float(np.sum((y.to_numpy() - y_hat) ** 2))
    total_sum_squares = float(np.sum((y.to_numpy() - y.mean()) ** 2))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares
    return r_squared, beta


def save_r_squared_html_chart(results: pd.DataFrame, output_path: Path) -> Path:
    chart_data = results.copy()
    chart_data["unexplained_variance"] = 1.0 - chart_data["r_squared"]

    fig = go.Figure(
        data=[
            go.Bar(
                name="R^2",
                x=chart_data["model"],
                y=chart_data["r_squared"],
                marker_color="#2ca25f",
                text=chart_data["r_squared"].map(lambda value: f"{value:.3f}"),
                textposition="outside",
                cliponaxis=False,
            ),
            go.Bar(
                name="Unexplained variance",
                x=chart_data["model"],
                y=chart_data["unexplained_variance"],
                marker_color="#de2d26",
            ),
        ]
    )
    fig.update_layout(
        title="Model R^2 and Unexplained Variance",
        barmode="stack",
        width=760,
        height=480,
        yaxis_title="Share of variance",
        yaxis=dict(range=[0, 1.08], tickformat=".0%"),
        xaxis_title="Model",
        legend_title_text="Component",
        template="plotly_white",
        font=dict(size=16),
        title_font=dict(size=22),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.24,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=120),
    )
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def save_r_squared_png_chart(results: pd.DataFrame, output_path: Path) -> Path:
    chart_data = results.copy()
    chart_data["unexplained_variance"] = 1.0 - chart_data["r_squared"]

    width = 760
    height = 480
    margin_left = 90
    margin_right = 35
    margin_top = 70
    margin_bottom = 100
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    green = "#2ca25f"
    red = "#de2d26"
    axis_color = "#333333"
    grid_color = "#dddddd"

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    title_font = ImageFont.load_default(size=24)
    label_font = ImageFont.load_default(size=20)

    draw.text((margin_left, 24), "Model R^2 and Unexplained Variance", fill=axis_color, font=title_font)

    for tick in np.linspace(0, 1, 6):
        y = margin_top + plot_height - int(tick * plot_height)
        draw.line((margin_left, y, width - margin_right, y), fill=grid_color, width=1)
        draw.text((35, y - 10), f"{tick:.0%}", fill=axis_color, font=label_font)

    draw.line((margin_left, margin_top, margin_left, margin_top + plot_height), fill=axis_color, width=2)
    draw.line(
        (margin_left, margin_top + plot_height, width - margin_right, margin_top + plot_height),
        fill=axis_color,
        width=2,
    )

    bar_count = len(chart_data)
    slot_width = plot_width / bar_count
    bar_width = min(130, slot_width * 0.55)

    for index, row in chart_data.iterrows():
        center_x = margin_left + slot_width * (index + 0.5)
        x0 = int(center_x - bar_width / 2)
        x1 = int(center_x + bar_width / 2)
        bottom = margin_top + plot_height
        r2_height = int(row["r_squared"] * plot_height)
        green_y0 = bottom - r2_height

        draw.rectangle((x0, green_y0, x1, bottom), fill=green)
        draw.rectangle((x0, margin_top, x1, green_y0), fill=red)

        r2_label = f'{row["r_squared"]:.3f}'
        r2_label_width = draw.textlength(r2_label, font=label_font)
        draw.text(
            (center_x - r2_label_width / 2, green_y0 - 26),
            r2_label,
            fill=axis_color,
            font=label_font,
        )

        label = str(row["model"])
        label_width = draw.textlength(label, font=label_font)
        draw.text(
            (center_x - label_width / 2, bottom + 22),
            label,
            fill=axis_color,
            font=label_font,
        )

    legend_y = height - 40
    draw.rectangle((margin_left, legend_y, margin_left + 24, legend_y + 24), fill=green)
    draw.text((margin_left + 34, legend_y + 3), "R^2", fill=axis_color, font=label_font)
    draw.rectangle((margin_left + 140, legend_y, margin_left + 164, legend_y + 24), fill=red)
    draw.text((margin_left + 174, legend_y + 3), "Unexplained variance", fill=axis_color, font=label_font)

    image.save(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    networks_dir = args.networks_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    networks = {
        name: load_network(resolve_network_file(networks_dir, file_options))
        for name, file_options in NETWORK_FILE_OPTIONS.items()
    }
    networks = align_networks(networks)
    predictors, target = flatten_networks(networks)

    model_specs = {
        "geo only": ["geo"],
        "industry only": ["industry"],
        "skill only": ["skill"],
        "all combined": ["geo", "industry", "skill"],
    }

    print(f"Regression target: occ_net transitions")
    print(f"Observations: {len(target):,} off-diagonal cells")
    print()

    results = []
    for model_name, columns in model_specs.items():
        r_squared, _ = fit_r_squared(predictors[columns], target)
        results.append({"model": model_name, "r_squared": r_squared})
        print(f"{model_name:14s} R^2 = {r_squared:.6f}")

    results_df = pd.DataFrame(results)
    html_chart_path = save_r_squared_html_chart(results_df, figures_dir / "r2_bar_chart.html")
    png_chart_path = save_r_squared_png_chart(results_df, figures_dir / "r2_bar_chart.png")
    print()
    print(f"Saved stacked R^2 HTML chart to {html_chart_path}")
    print(f"Saved stacked R^2 PNG chart to {png_chart_path}")


if __name__ == "__main__":
    main()
