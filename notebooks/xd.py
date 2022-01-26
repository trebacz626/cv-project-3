if __name__ == "__main__":
    from torch_lr_finder import LRFinder
    from torch import nn
    model = get_model_instance_segmentation(23)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_finder = LRFinder(model, optimizer, criterion, device="cpu")
    lr_finder.range_test(dataset_loader_test, end_lr=100, num_iter=100)
    lr_finder.plot()  # to inspect the loss-learning rate graph
    lr_finder.reset()
