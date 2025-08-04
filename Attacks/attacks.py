import torch
import torch.nn as nn

class Attack:
    """
    Base class for all attacks, establishing a consistent interface.
    """
    def __init__(self, model):
        # The model passed here is the original SNN model.
        self.model = model
        self.device = next(model.parameters()).device

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        # This makes the class callable like a function: attack(images, target)
        return self.forward(*args, **kwargs)


"""Adding all the attacks which I shall use now"""

class FGSM(Attack):
    """
    FGSM attack specialized for the SNN regression model.
    This class knows that the model returns a tuple (outputs, cost, counts).
    """
    def __init__(self, model, eps=0.03):
        super().__init__(model)
        self.eps = eps
        self.loss = nn.MSELoss()

    def forward(self, images, target):
        images = images.clone().detach().to(self.device)
        target = target.clone().detach().to(self.device)
        
        images.requires_grad = True
            
        # Get the tuple output directly from the SNN model
        outputs, _, _ = self.model(images)
        final_prediction = outputs.mean()
        
        # Calculate loss
        cost = self.loss(final_prediction, target.squeeze())
        
        # Get the gradient of the loss with respect to the input
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]
        
        # Create the adversarial image
        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=-1, max=1).detach()
        
        return adv_images

class PGD(Attack):
    """
    PGD attack specialized for the SNN regression model.
    This class knows that the model returns a tuple (outputs, cost, counts).
    """
    def __init__(self, model, eps=0.03, alpha=2/255, steps=10, random_start=True):
        super().__init__(model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.loss = nn.MSELoss()

    def forward(self, images, target):
        images = images.clone().detach().to(self.device)
        target = target.clone().detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Start from a random point within the epsilon ball
            delta = torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images + delta, min=-1, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            
            # Get the tuple output directly from the SNN model
            outputs, _, _ = self.model(adv_images)
            final_prediction = outputs.mean()
            
            # Calculate loss
            cost = self.loss(final_prediction, target.squeeze())
            
            # Get gradient of the loss
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            # Perform the PGD step
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=-1, max=1).detach()

        return adv_images


class MIFGSM(Attack):
    """
    Momentum Iterative FGSM (MIFGSM) attack specialized for a regression model.
    This attack incorporates momentum into the iterative process for more effective attacks.
    """
    def __init__(self, model, eps=0.03, alpha=2/255, steps=10, decay=1.0):
        super().__init__(model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.loss = nn.MSELoss()

    def forward(self, images, target):
        images = images.clone().detach().to(self.device)
        target = target.clone().detach().to(self.device)

        adv_images = images.clone().detach()
        momentum = torch.zeros_like(images).detach().to(self.device)

        for _ in range(self.steps):
            adv_images.requires_grad = True
            
            # Get the tuple output directly from the SNN model
            outputs, _, _ = self.model(adv_images)
            final_prediction = outputs.mean()
            
            # Calculate loss
            cost = self.loss(final_prediction, target.squeeze())
            
            # Get gradient of the loss
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            
            # Calculate momentum gradient
            grad_norm = torch.norm(grad, p=1)
            grad = grad / grad_norm
            grad = grad + self.decay * momentum
            momentum = grad
            
            # Perform the MIFGSM step
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=-1, max=1).detach()

        return adv_images