import pandas as pd
import torch
import baseline
import cvae
from util import get_data, visualize, generate_table


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    results = []
    columns = []

    for num_quadrant_inputs in [1, 2, 3]:
        maybes = 's' if num_quadrant_inputs > 1 else ''
        print(f'Training with {num_quadrant_inputs} quadrant{maybes} as input...')

        # Dataset
        datasets, dataloaders, dataset_sizes = get_data(
            num_quadrant_inputs=num_quadrant_inputs,
            batch_size=128
        )

        # Train baseline
        baseline_net = baseline.train(
            device=device,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            learning_rate=1e-3,
            num_epochs=num_epochs,
            early_stop_patience=3,
            model_path=f'baseline_net_q{num_quadrant_inputs}.pth'
        )

        # Train CVAE
        cvae_net = cvae.train(
            device=device,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            learning_rate=1e-3,
            num_epochs=num_epochs,
            early_stop_patience=3,
            model_path=f'cvae_net_q{num_quadrant_inputs}.pth',
            pre_trained_baseline_net=baseline_net
        )

        # Visualize conditional predictions
        visualize(
            device=device,
            num_quadrant_inputs=num_quadrant_inputs,
            pre_trained_baseline=baseline_net,
            pre_trained_cvae=cvae_net,
            num_images=10,
            num_samples=10,
            image_path=f'cvae_plot_q{num_quadrant_inputs}.png'
        )

        # Retrieve conditional log likelihood
        df = generate_table(
            device=device,
            num_quadrant_inputs=num_quadrant_inputs,
            pre_trained_baseline=baseline_net,
            pre_trained_cvae=cvae_net,
            num_particles=10,
            col_name=f'{num_quadrant_inputs} quadrant{maybes}'
        )
        results.append(df)
        columns.append(f'{num_quadrant_inputs} quadrant{maybes}')

    results = pd.concat(results, axis=1, ignore_index=True)
    results.columns = columns
    results.loc['Performance gap', :] = results.iloc[0, :] - results.iloc[1, :]

    results.to_csv('results.csv')
