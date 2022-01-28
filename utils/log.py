import json
from utils.image_gen import segmentation_to_image, attention_to_images, shape_attention

def tb_log(train_metrics,val_metrics,writer,attribute, epoch):
    for key,metric_hash in train_metrics.items():
        writer.add_scalars(f'{attribute}/Training/{key}', metric_hash, epoch)
    for key,metric_hash in val_metrics.items():
        writer.add_scalars(f'{attribute}/Val/{key}', metric_hash, epoch)

def console_log(train_metrics,val_metrics, epoch, step=None):
    print(f"Results - Epoch: {epoch} - Step: {step}")
    print(json.dumps(train_metrics, indent=2))
    if val_metrics:
        print(f"Validation Results - Epoch: {epoch}")
        print(json.dumps(val_metrics, indent=2))

def comet_log(metrics, experiment, epoch=None, step=None):
    experiment.log_metrics(metrics, epoch=epoch, step=step)

def comet_image_log(image,image_name,experiment,epoch=None):
    experiment.log_image(image, name=image_name)

def image_log(segmentation,original,attention_map,palette,experiment,epoch, normalize='local', dim=None):
    if segmentation is not None:
        seg_img = segmentation_to_image(segmentation, original, palette)
        comet_image_log(seg_img,f'segmentation_epoch:{epoch}',experiment, epoch=epoch)
    if attention_map is not None:
        attentions, _ = attention_to_images(original, shape_attention(attention_map, dim=dim), normalize=normalize)
        for i,image in enumerate(attentions):
            comet_image_log(image,f'attention_head:{i}_epoch:{epoch}',experiment, epoch=epoch)
    comet_image_log(original,f'original_epoch{epoch}',experiment, epoch=epoch)