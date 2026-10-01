import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def plot_batch_metrics(
    current_batch_data,
    training_state,
    training_config,
    i,
):
    """Handles plotting and printing of metrics for a single batch."""
    
    # Unpack state and config
    fractional_epochs = training_state['fractional_epochs']
    losses = training_state['losses']
    pos_losses = training_state['pos_losses']
    normals_regs = training_state['normals_regs']
    distortion_regs = training_state['distortion_regs']
    times = training_state['times']
    optimizer = training_state['optimizer']
    
    normals_reg_coeff = training_config['normals_reg_coeff']
    distortion_reg_coeff = training_config['distortion_reg_coeff']

    # Unpack current batch data
    batch_loss = current_batch_data['batch_loss']
    position_loss = current_batch_data['position_loss']
    normals_reg = current_batch_data['normals_reg']
    distortion_reg = current_batch_data['distortion_reg']
    cur_udf = current_batch_data['cur_udf']
    batch_time = current_batch_data['batch_time']
    total_batch_time = current_batch_data['total_batch_time']
    
    try:
        plt.close('all') 
    except:
        pass
        
    clear_output(wait=True)

    # Plot losses
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    axs[0].plot(fractional_epochs, np.log10(losses), label='loss')
    axs[0].plot(fractional_epochs, np.log10(pos_losses), label='pos loss')
    if normals_reg_coeff > 0:
        axs[0].plot(fractional_epochs, np.log10(normals_regs), label='normals reg')
    if distortion_reg_coeff > 0:
        axs[0].plot(fractional_epochs, np.log10(distortion_regs), label='distortion reg')
        
    axs[0].legend()
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Log10 of Loss')
    axs[0].set_title('Log10 of Losses')

    # Write current losses to the plot in text
    def add_text(ax, y_pos, label, value):
        ax.text(
            0.8, y_pos, f"{label}: {value:.8f}",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, color='black',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

    add_text(axs[0], 0.9, "Batch Loss", batch_loss.item())
    add_text(axs[0], 0.8, "Batch Pos. Loss", position_loss.item())

    if normals_reg_coeff > 0:
        add_text(axs[0], 0.7, "Batch Normals Reg.", normals_reg.item())
    if distortion_reg_coeff > 0:
        add_text(axs[0], 0.6, "Batch Distortion Reg.", distortion_reg.item())

    axs[1].plot(fractional_epochs, times)
    axs[1].set_xlabel('Epoch')
    axs[1].set_title('Time per Batch')
    add_text(axs[1], 0.90, "Batch Time", batch_time)

    plt.tight_layout()
    display(fig)
    plt.close(fig) 

    # Print out extra info below the plots
    if normals_reg_coeff > 0:
        print(f"Batch {i}, Loss: {batch_loss:.8f}, Pos. Loss: {position_loss:.8f}, "
              f"Normals Reg: {(normals_reg_coeff * normals_reg):.8f}, Batch Time: {batch_time:.4f}s")
    else:
        print(f"Batch {i}, Loss: {batch_loss:.8f}, Pos. Loss: {position_loss:.8f}, Batch Time: {batch_time:.4f}s")
    
    print(f"mean UDF: {cur_udf.mean():.8f}")
    print(f"Time inc plotting: {total_batch_time:.4f}s")
    
    primary_lr = optimizer.param_groups[0]['lr']
    print(f"Current Learning Rate: {primary_lr:.8f}")


def handle_epoch_end(epoch_loss, total_num_points, bcs, epoch_print_rate, epoch_save_rate, current_epoch):
    """Handles logic at the end of each epoch (loss update, printing, saving)."""

    # Calculate and update loss
    loss = torch.tensor(epoch_loss / total_num_points)

    if current_epoch % epoch_print_rate == 0:
        print(f"Epoch {current_epoch}, Loss: {loss.item():.8f}")

    if current_epoch % epoch_save_rate == 0:
        #bcs.save_mlps('models/temp.pth')
        bcs.save_poly_coeffs('models/temp.pth')
        print('Saved checkpoint.')
        
    # Returns the new loss value and incremented epoch
    return loss, current_epoch + 1