from shared.configuration_classes import FeatureConfiguration
from shared.enums import FeatureEnum

default = FeatureConfiguration('processed', 'graphsage', 'kg', False, FeatureEnum.ENTITIES)

graphsage = FeatureConfiguration('graphsage', 'graphsage', 'kg', False, FeatureEnum.ENTITIES,
                                      attributes=['name', 'description'])
graphsage_scale = FeatureConfiguration('graphsage_scale', 'graphsage', 'kg', False, FeatureEnum.ENTITIES,
                                      attributes=['name', 'description'], scale=True)
graphsage_item = FeatureConfiguration('graphsage_item', 'graphsage', 'kg', False, FeatureEnum.ITEMS, scale=True,
                                      attributes=['name', 'description'])
complEX = FeatureConfiguration('complex', 'complex', 'kg', False, FeatureEnum.DESC_ENTITIES, model_name='complex')
comsage = FeatureConfiguration('comsage', None, 'kg', False, FeatureEnum.ENTITIES, scale=True)
idcf = FeatureConfiguration('key_user', 'idcf', 'cg_pos', True, FeatureEnum.USERS)

feature_configurations = [graphsage, graphsage_item, default, idcf, complEX, comsage]
feature_conf_names = [f.name for f in feature_configurations]