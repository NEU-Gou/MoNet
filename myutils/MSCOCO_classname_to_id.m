function id = MSCOCO_classname_to_id(classname)
    global classnames;
    global classids;
    
    id = [];
    for i=1:numel(classnames)
        if(strcmp(classname , classnames{i}))
            id = classids(i);
            break;
        end
    end
    
    if(isempty(id))
        error('no such classname');
    end
end