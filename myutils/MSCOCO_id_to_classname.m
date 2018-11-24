function classname = MSCOCO_id_to_classname(id)
    global classids;
    global classnames;
    
    classname = '';
    for i=1:numel(classnames)
        if(id==classids(i))
            classname = classnames{i};
            break;
        end
    end    
end