#!/usr/bin/env python3
"""
reid-strong-baseline 모델을 TAO 호환 형식으로 변환하는 최종 스크립트
모든 의존성 모듈을 완전히 재구성합니다.
"""

import torch
import torch.nn as nn
import os
import sys
import types
from collections import OrderedDict

def create_complete_modeling_structure():
    """완전한 modeling 모듈 구조를 생성합니다."""
    
    # modeling 패키지 생성
    modeling_pkg = types.ModuleType('modeling')
    modeling_pkg.__path__ = ['modeling']  # 패키지로 만들기 위해 필요
    
    # modeling.backbones 패키지 생성
    backbones_pkg = types.ModuleType('modeling.backbones')
    backbones_pkg.__path__ = ['modeling/backbones']
    
    # ResNet 관련 클래스들
    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out

    class Bottleneck(nn.Module):
        expansion = 4
        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out

    class ResNet(nn.Module):
        def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
            self.inplanes = 64
            super(ResNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x

        def load_param(self, model_path):
            pass  # 더미 구현

    # SENet 관련 더미 클래스들
    class SEResNetBottleneck(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            
    class SEBottleneck(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            
    class SEResNeXtBottleneck(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
    
    class SENet(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            
        def load_param(self, model_path):
            pass

    def resnet50_ibn_a(last_stride):
        return ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3])

    # modeling.backbones.resnet 모듈
    resnet_module = types.ModuleType('modeling.backbones.resnet')
    resnet_module.ResNet = ResNet
    resnet_module.BasicBlock = BasicBlock
    resnet_module.Bottleneck = Bottleneck
    
    # modeling.backbones.senet 모듈
    senet_module = types.ModuleType('modeling.backbones.senet')
    senet_module.SENet = SENet
    senet_module.SEResNetBottleneck = SEResNetBottleneck
    senet_module.SEBottleneck = SEBottleneck
    senet_module.SEResNeXtBottleneck = SEResNeXtBottleneck
    
    # modeling.backbones.resnet_ibn_a 모듈
    resnet_ibn_a_module = types.ModuleType('modeling.backbones.resnet_ibn_a')
    resnet_ibn_a_module.resnet50_ibn_a = resnet50_ibn_a

    # Baseline 모델 클래스
    class Baseline(nn.Module):
        in_planes = 2048
        
        def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
            super(Baseline, self).__init__()
            
            if model_name == 'resnet18':
                self.in_planes = 512
                self.base = ResNet(last_stride=last_stride, block=BasicBlock, layers=[2, 2, 2, 2])
            elif model_name == 'resnet34':
                self.in_planes = 512
                self.base = ResNet(last_stride=last_stride, block=BasicBlock, layers=[3, 4, 6, 3])
            elif model_name == 'resnet50':
                self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3])
            elif model_name == 'resnet101':
                self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 23, 3])
            elif model_name == 'resnet152':
                self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 8, 36, 3])
            else:
                # 기본값
                self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3])

            self.gap = nn.AdaptiveAvgPool2d(1)
            self.num_classes = num_classes
            self.neck = neck
            self.neck_feat = neck_feat

            if self.neck == 'no':
                self.classifier = nn.Linear(self.in_planes, self.num_classes)
            elif self.neck == 'bnneck':
                self.bottleneck = nn.BatchNorm1d(self.in_planes)
                self.bottleneck.bias.requires_grad_(False)
                self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        def forward(self, x):
            global_feat = self.gap(self.base(x))
            global_feat = global_feat.view(global_feat.shape[0], -1)
            if self.neck == 'no':
                feat = global_feat
            elif self.neck == 'bnneck':
                feat = self.bottleneck(global_feat)
            if self.training:
                cls_score = self.classifier(feat)
                return cls_score, global_feat
            else:
                if self.neck_feat == 'after':
                    return feat
                else:
                    return global_feat

        def load_param(self, trained_path):
            param_dict = torch.load(trained_path)
            for i in param_dict:
                if 'classifier' in i:
                    continue
                self.state_dict()[i].copy_(param_dict[i])

    # modeling.baseline 모듈
    baseline_module = types.ModuleType('modeling.baseline')
    baseline_module.Baseline = Baseline
    
    # 모든 모듈을 패키지에 연결
    backbones_pkg.resnet = resnet_module
    backbones_pkg.senet = senet_module
    backbones_pkg.resnet_ibn_a = resnet_ibn_a_module
    
    modeling_pkg.backbones = backbones_pkg
    modeling_pkg.baseline = baseline_module
    
    return modeling_pkg, backbones_pkg, resnet_module, senet_module, resnet_ibn_a_module, baseline_module

def register_modules():
    """필요한 모든 모듈을 sys.modules에 등록합니다."""
    
    modeling_pkg, backbones_pkg, resnet_module, senet_module, resnet_ibn_a_module, baseline_module = create_complete_modeling_structure()
    
    # sys.modules에 등록
    sys.modules['modeling'] = modeling_pkg
    sys.modules['modeling.backbones'] = backbones_pkg
    sys.modules['modeling.backbones.resnet'] = resnet_module
    sys.modules['modeling.backbones.senet'] = senet_module
    sys.modules['modeling.backbones.resnet_ibn_a'] = resnet_ibn_a_module
    sys.modules['modeling.baseline'] = baseline_module
    
    print("✓ Registered all modeling modules")

def cleanup_modules():
    """등록된 모듈들을 정리합니다."""
    modules_to_remove = [
        'modeling',
        'modeling.backbones',
        'modeling.backbones.resnet',
        'modeling.backbones.senet', 
        'modeling.backbones.resnet_ibn_a',
        'modeling.baseline'
    ]
    
    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]

def load_reid_model_safe(model_path):
    """
    reid-strong-baseline 모델을 안전하게 로드합니다.
    """
    print(f"Loading reid-strong-baseline model from: {model_path}")
    
    # 모듈 등록
    register_modules()
    
    try:
        # 모델 로드
        checkpoint = torch.load(model_path, map_location='cpu')
        print("✓ Successfully loaded reid-strong-baseline model")
        
        # state_dict 추출
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state_dict = checkpoint
            else:
                # 다른 키들 확인
                for key in ['model', 'net', 'model_state_dict']:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        break
                else:
                    raise ValueError("Could not find state_dict in checkpoint")
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        
        return state_dict, checkpoint
        
    finally:
        # 모듈 정리
        cleanup_modules()

def convert_reid_to_tao(source_model_path, target_model_path, num_classes=751):
    """
    reid-strong-baseline 모델을 TAO 호환 형식으로 변환
    """
    
    # reid 모델 로드
    state_dict, original_checkpoint = load_reid_model_safe(source_model_path)
    
    print(f"Original model has {len(state_dict)} parameters")
    
    # 처음 몇 개 파라미터 이름 출력
    print("\nOriginal parameter names (first 10):")
    for i, (key, value) in enumerate(list(state_dict.items())[:10]):
        print(f"  {i+1}. {key}: {value.shape}")
    
    # TAO 호환 형식으로 키 변환
    tao_state_dict = OrderedDict()
    
    # reid-strong-baseline의 키 구조를 TAO 형식으로 매핑
    # TAO는 model.base.*를 기대함 (model.backbone.* 아님)
    key_mappings = [
        # Backbone 매핑 - TAO는 model.base를 사용
        ('base.', 'model.base.'),
        
        # Bottleneck/Neck 매핑 
        ('bottleneck.', 'model.bottleneck.'),
        
        # Classifier 매핑 - 크기 불일치로 제외
        # ('classifier.', 'model.classifier.'),
        
        # GAP 매핑 (필요한 경우)
        ('gap.', 'model.gap.'),
    ]
    
    converted_count = 0
    skipped_count = 0
    for old_key, value in state_dict.items():
        new_key = old_key
        
        # 특별 처리: classifier.weight 크기 조정
        if old_key == 'classifier.weight':
            # reid-strong-baseline: [751, 2048] -> TAO: [751, 2048] (feat_dim=2048이므로 그대로 사용 가능)
            print(f"Processing classifier layer: {old_key} with shape {value.shape}")
            new_key = 'model.classifier.weight'
            tao_state_dict[new_key] = value
            converted_count += 1
            continue
        
        # 키 매핑 적용
        for old_pattern, new_pattern in key_mappings:
            if old_key.startswith(old_pattern):
                new_key = old_key.replace(old_pattern, new_pattern, 1)
                converted_count += 1
                break
        
        tao_state_dict[new_key] = value
        
        # 처음 몇 개만 출력
        if len(tao_state_dict) <= 5 and old_key != new_key:
            print(f"Mapped: {old_key} -> {new_key}")
    
    print(f"\nConverted {converted_count} parameter names")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} parameters")
    
    # TAO 형식의 체크포인트 생성
    tao_checkpoint = {
        'state_dict': tao_state_dict,
        'epoch': original_checkpoint.get('epoch', 120) if isinstance(original_checkpoint, dict) else 120,
        'global_step': original_checkpoint.get('global_step', 0) if isinstance(original_checkpoint, dict) else 0,
        'pytorch-lightning_version': '1.9.0',
        'hyper_parameters': {
            'experiment_spec': {
                'model': {
                    'backbone': 'resnet_50',
                    'num_classes': num_classes,
                    'feat_dim': 256,
                    'neck': 'bnneck',
                    'metric_loss_type': 'triplet',
                    'with_center_loss': False,
                    'label_smooth': True,
                    'last_stride': 1,
                    'neck_feat': 'after'
                },
                'dataset': {
                    'num_classes': num_classes
                }
            },
            'prepare_for_training': False
        }
    }
    
    # 변환된 모델 저장
    os.makedirs(os.path.dirname(target_model_path), exist_ok=True)
    torch.save(tao_checkpoint, target_model_path)
    
    # 저장된 파일 크기 확인
    file_size = os.path.getsize(target_model_path) / (1024 * 1024)  # MB
    
    print(f"\n✓ Converted model saved to: {target_model_path}")
    print(f"✓ File size: {file_size:.2f} MB")
    print(f"✓ Total parameters converted: {len(tao_state_dict)}")
    
    return True

def main():
    """메인 함수"""
    if len(sys.argv) != 3:
        print("Usage: python convert_reid_to_tao_final.py <source_model_path> <target_model_path>")
        print("Example: python convert_reid_to_tao_final.py /path/to/reid_model.pth /path/to/tao_model.pth")
        sys.exit(1)
    
    source_path = sys.argv[1]
    target_path = sys.argv[2]
    
    if not os.path.exists(source_path):
        print(f"Error: Source model file not found: {source_path}")
        sys.exit(1)
    
    try:
        convert_reid_to_tao(source_path, target_path)
        print("\n🎉 Reid-strong-baseline model conversion completed successfully!")
        print(f"You can now use the converted model with TAO: {target_path}")
        
        print("\nNext steps:")
        print("1. Use the converted model in TAO evaluation:")
        print(f"   evaluate.checkpoint={target_path}")
        print("2. Make sure your TAO config file matches the original training settings")
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
