if __package__:
    from .models.state_hijacking_dit import StateInjectionDiTRELAY
else:
    from models.state_hijacking_dit import StateInjectionDiTRELAY

__all__ = ["StateInjectionDiTRELAY"]
